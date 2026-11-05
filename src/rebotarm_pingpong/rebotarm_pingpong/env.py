"""使用笛卡尔拍面控制、自由旋转球体和双目 RGB 观测的训练环境。"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml

from rebotarm_description.parameters import description_path
from rebotarm_description.pingpong_model import validate_config
from .control import CartesianServo, VerticalBaseline, horizontal_home
from .vision import CameraBallObserver


class PingPong1DEnv(gym.Env):
    """只用相机观测的自由球环境，策略学习笛卡尔控制残差。

    球的真值仅供物理积分、重置和独立计分使用；所有球特征和基线指令都来自
    渲染的 RGB 图像。类名保留 ``PingPong1DEnv`` 以兼容旧的启动脚本和导入。
    当前动作是五维的 XYZ 拍面速度及两个局部倾角速度；传入单元素动作时，
    将它当作旧版竖直残差，方便迁移旧调用方。
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    observation_version = "stereo_cartesian_free_ball_v2"

    def __init__(self, *, randomize=False, render_mode=None, config_path=None):
        # 环境初始化：加载统一的机器人参数、乒乓球场景和相机配置。
        # 场景仍使用自由球体；控制器只能通过相机观测得到球的信息。
        root = description_path()
        self.config = yaml.safe_load(Path(config_path or root / "config/pingpong.yaml").read_text())
        validate_config(self.config)
        from rebotarm_description.pingpong_model import derived_pingpong_files
        from rebotarm_description.parameters import load_hardware, joint_limits
        limits = joint_limits(root)
        files = derived_pingpong_files((root / "mjcf/robot.xml").read_text(), self.config,
                                      load_hardware(root / "config/rebotarm_hardware.yaml"), limits)
        if config_path:
            assets = {p.name: p.read_bytes() for p in (root / "meshes").glob("*.stl")}
            assets.update({name: value.encode() for name, value in files.items()})
            self.model = mujoco.MjModel.from_xml_string(files["pingpong_3d_scene.xml"], assets)
        else:
            if any(not (root / "mjcf" / name).is_file() or (root / "mjcf" / name).read_text() != value for name, value in files.items()):
                raise ValueError("pingpong generated models are stale; run python3 scripts/sync_robot_parameters.py")
            self.model = mujoco.MjModel.from_xml_path(str(root / "mjcf/pingpong_3d_scene.xml"))
        for name in (f"joint{i}" for i in range(1, 7)):
            if not np.allclose(self.model.joint(name).range, [limits[name]["lower"], limits[name]["upper"]], atol=1e-12, rtol=0):
                raise ValueError("shared arm limits are stale; run python3 scripts/sync_robot_parameters.py")
        self.data = mujoco.MjData(self.model)
        self.home_q = horizontal_home(self.model, self.config)
        self.baseline, self.servo = VerticalBaseline(self.config), CartesianServo(self.model, self.config)
        perception = dict(self.config["perception"], ball_radius_m=self.config["ball"]["radius_m"])
        self.observer = CameraBallObserver(self.model, self.config["cameras"], perception,
                                           gravity=self.config["scene"]["gravity_m_s2"])
        self.randomize, self.render_mode = randomize, render_mode
        # 动作为 [vx, vy, vz, 拍面横滚角速度, 拍面俯仰角速度] 残差。
        # 规则基线负责基本抛物线运动，PPO 只学习有限幅度的修正。
        self.action_space = gym.spaces.Box(-1., 1., (5,), dtype=np.float32)
        # 观测包括：相机估计的球相对位置/速度、拍面状态、规则基线指令、
        # 六个关节编码器、上一动作和视觉跟踪质量；不直接包含仿真真值。
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32)
        self.ball_body = self.model.body("pingpong_ball").id
        self.ball_geom = self.model.geom("ball").id
        self.paddle_geom = self.model.geom("paddle_face").id
        self.site = self.model.site("paddle_surface").id
        joint = self.model.joint("pingpong_ball_freejoint")
        self.ball_q, self.ball_dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.frame_skip = round(1 / (self.config["control"]["policy_hz"] * self.model.opt.timestep))
        self.servo_skip = round(1 / (self.config["control"]["servo_hz"] * self.model.opt.timestep))
        self.nominal_mass = self.model.body_mass[self.ball_body]
        self.nominal_inertia = self.model.body_inertia[self.ball_body].copy()
        self.nominal_pair_solref = self.model.pair_solref[0].copy()
        self.nominal_pair_friction = self.model.pair_friction[0].copy()
        self.renderer = None
        self.last_action = np.zeros(5, dtype=np.float32)
        self.baseline_command = np.zeros(5, dtype=float)
        self._done = True

    def truth_ball_state(self):
        """仅供仿真统计使用的球位置和速度，不会提供给控制策略。"""
        return self.data.xpos[self.ball_body].copy(), self.data.qvel[self.ball_dof:self.ball_dof+3].copy()

    def state(self):
        """兼容旧接口的竖直方向真值，只有诊断用途，控制器不会读取。"""
        position, velocity = self.truth_ball_state()
        paddle, pv, _ = self.servo.paddle_state(self.data)
        return float(position[2]), float(velocity[2]), float(paddle[2]), float(pv[2])

    def _observation(self):
        # 相机无效时不回退到球真值，而是让机械臂回到安全行程并等待视觉恢复。
        estimate = self.observer.observe(self.data)
        paddle, pv, rotation = self.servo.paddle_state(self.data)
        if estimate.valid:
            relative = estimate.position - paddle
            relative[2] -= self.config["ball"]["radius_m"]
            velocity = estimate.velocity
            command = self._camera_baseline(estimate, paddle, pv)
        else:
            # 不使用真值回退；无效标志会进入观测，执行器暂时收回行程。
            relative = np.zeros(3)
            velocity = np.zeros(3)
            command = np.array([0., 0., np.clip(8*(self.baseline.home_z-paddle[2]), -.35, .35), 0., 0.])
        self.baseline_command = np.asarray(command, dtype=float)
        # 仅为诊断和只查看竖直分量的旧调用方保留；它不会替代下面的 XYZ 命令。
        self.baseline_speed = float(self.baseline_command[2])
        age = min(1., max(0., float(estimate.age_s)))
        latency = max(0., float(estimate.received_timestamp_s-estimate.measurement_timestamp_s))
        if not np.isfinite(latency):
            latency = 0.
        command_scale = np.r_[self._residual_velocity_scale(), self._residual_tilt_scale()]
        tilt_scale = max(float(self.servo.tilt_limit_rad), 1e-6)
        # 观测维度依次为 3+3+3+3+3+2+5+6+6+5+4 = 43；最后四项包含
        # 视觉有效标志、置信度、估计年龄和端到端延迟。
        return np.r_[relative/[.1, .1, .2], velocity/2.,
                     (paddle-self.servo.home_xyz)/[.1, .1, .05], pv/[.5, .5, .6],
                     rotation[:, 2], self.servo.cartesian_tilt/tilt_scale,
                     self.baseline_command/command_scale, self.data.qpos[:6]/np.pi,
                     self.data.qvel[:6]/5., self.last_action,
                     float(estimate.valid), estimate.confidence, age/.1, latency/.1].astype(np.float32)

    def _residual_velocity_scale(self):
        values = np.asarray(self.config["control"].get("residual_cartesian_speed_m_s", [.20, .20, .20]), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("residual_cartesian_speed_m_s must be a finite positive XYZ vector")
        return values

    def _residual_tilt_scale(self):
        values = np.asarray(self.config["control"].get("residual_tilt_rate_rad_s", [.8, .8]), dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("residual_tilt_rate_rad_s must be a finite positive XY vector")
        return values

    def _camera_baseline(self, estimate, paddle, paddle_velocity):
        """只根据双目估计生成有速度和倾角限幅的笛卡尔跟踪指令。"""
        target_and_tilt = self.servo.target_from_ball_estimate(
            estimate, lookahead_s=float(self.config["control"].get("tracking_lookahead_s", .06)),
            target_z=self.servo.reference[2])
        if target_and_tilt is None:
            return np.array([0., 0., np.clip(8*(self.baseline.home_z-paddle[2]), -.35, .35), 0., 0.])
        target, tilt_target = target_and_tilt
        c = self.config["control"]
        xy_gain = float(c.get("xy_tracking_gain_s", 3.0))
        xy_damping = float(c.get("xy_tracking_damping", .45))
        xy = xy_gain*(target[:2]-paddle[:2]) - xy_damping*paddle_velocity[:2]
        xy = np.clip(xy, -self.servo.cartesian_speed_limit[:2], self.servo.cartesian_speed_limit[:2])
        vz = self.baseline.command(estimate.position[2], estimate.velocity[2], paddle[2], paddle_velocity[2])
        tilt_time = max(float(c.get("tilt_tracking_time_s", .08)), 1e-3)
        tilt_rate = (np.asarray(tilt_target)-self.servo.cartesian_tilt)/tilt_time
        tilt_rate = np.clip(tilt_rate, -self.servo.tilt_rate_limit, self.servo.tilt_rate_limit)
        return np.r_[xy, vz, tilt_rate]

    def _record_vision_metrics(self):
        """记录离线视觉误差；这些误差只用于评估，不会反馈给策略。"""
        estimate = self.observer.estimate
        self.vision_samples += 1
        if estimate.valid:
            self.last_visual_valid_time = float(self.data.time)
            self.valid_vision_samples += 1
            position, velocity = self.truth_ball_state()
            self.position_errors.append(float(np.linalg.norm(estimate.position-position)))
            self.velocity_errors.append(float(np.linalg.norm(estimate.velocity-velocity)))
            self.frame_ages.append(float(estimate.age_s))
            if not self.in_contact and self.data.time-self.last_bounce_time > .08:
                self.flight_position_errors.append(self.position_errors[-1])
                self.flight_velocity_errors.append(self.velocity_errors[-1])

    def reset(self, *, seed=None, options=None):
        """重置物理、相机跟踪器和统计量；``options`` 可固定初始球状态。"""
        # 每回合恢复球的质量、碰撞参数和随机初始状态；球仍可自由平移和旋转。
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        r, e = self.config["experiment"]["randomization"], self.config["experiment"]
        scale = self.np_random.uniform(*r["mass_scale"]) if self.randomize else 1.
        self.model.body_mass[self.ball_body] = self.nominal_mass * scale
        self.model.body_inertia[self.ball_body] = self.nominal_inertia * scale
        self.model.pair_solref[0] = self.nominal_pair_solref
        self.model.pair_friction[0] = self.nominal_pair_friction
        if self.randomize:
            self.model.pair_solref[0, 0] *= self.np_random.uniform(*r["contact_time_scale"])
            self.model.pair_solref[0, 1] = self.np_random.uniform(*r["damping_ratio"])
            self.model.pair_friction[0] *= self.np_random.uniform(*r["friction_scale"])
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:6] = self.home_q
        height = options.get("height_m", self.np_random.uniform(*e["initial_height_m"]))
        velocity = options.get("velocity_m_s", self.np_random.uniform(*e["initial_velocity_m_s"]))
        position = np.array(self.config["scene"]["home_surface_xyz_m"], dtype=float)
        position[:2] += np.asarray(options.get("xy_offset_m", self.np_random.uniform(*e["initial_xy_offset_m"], size=2)))
        position[2] += self.config["ball"]["radius_m"] + height
        self.data.qpos[self.ball_q:self.ball_q+3] = position
        self.data.qpos[self.ball_q+3:self.ball_q+7] = [1., 0., 0., 0.]
        self.data.qvel[self.ball_dof:self.ball_dof+2] = options.get("horizontal_velocity_m_s", self.np_random.uniform(*e["initial_horizontal_velocity_m_s"], size=2))
        self.data.qvel[self.ball_dof+2] = velocity
        self.data.qvel[self.ball_dof+3:self.ball_dof+6] = options.get("spin_rad_s", self.np_random.uniform(*e["initial_spin_rad_s"], size=3))
        self.actuator_gain = self.np_random.uniform(*r["actuator_gain"]) if self.randomize else 1.
        self.wind = np.array(self.config["scene"].get("wind_velocity_m_s", [0., 0., 0.]), dtype=float)
        if self.randomize:
            self.wind += self.np_random.uniform(*r.get("wind_m_s", [-.1, .1]), size=3)
        self.servo.reset()
        self.observer.reset(self.np_random)
        self.bounces = self.steps = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        self.in_contact = False
        self.contact_started = self.contact_vin = 0.
        self.last_bounce_time = -1.
        self.apex_heights = []
        self.bounce_radial_errors = []
        self.horizontal_errors = []
        self.last_apex_v = velocity
        self.max_torque_ratio = self.max_joint_speed = 0.
        self.position_errors, self.velocity_errors, self.frame_ages = [], [], []
        self.flight_position_errors, self.flight_velocity_errors = [], []
        self.torque_square_integral = np.zeros(6)
        self.vision_samples = self.valid_vision_samples = 0
        self.last_visual_valid_time = 0.
        self.termination = ""
        self._done = False
        mujoco.mj_forward(self.model, self.data)
        observation = self._observation()
        self._record_vision_metrics()
        return observation, self._info()

    def _contact(self):
        # 判断球是否真正击中拍面中心。拍框/侧面碰撞仍是物理碰撞，
        # 但会标记为失球，不能算作一次成功颠球。
        # 每个 0.125 ms 物理步都会调用这里。用整数比较判断无序的球/拍面几何
        # 对，避免每次查询都创建集合和生成器。
        self.edge_contact = False
        face_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if ((first == self.ball_geom and second == self.paddle_geom) or
                    (first == self.paddle_geom and second == self.ball_geom)):
                # 拍框或底面撞击仍是真实物理碰撞，但不能算作橡胶上表面的颠球。
                rotation = self.data.site_xmat[self.site].reshape(3, 3)
                point = rotation.T @ (contact.pos-self.data.site_xpos[self.site])
                normal_alignment = abs(float(np.dot(contact.frame[:3], rotation[:, 2])))
                on_top = (abs(point[2]) < .0015 and normal_alignment > .8 and
                          np.linalg.norm(point[:2]) < self.config["paddle"]["radius_m"]-.0002)
                face_contact |= on_top
                self.edge_contact |= not on_top
        return face_contact

    def apply_ball_aerodynamics(self):
        """施加世界坐标系空气阻力、马格努斯升力和自旋衰减力矩。"""
        b, start = self.config["ball"], self.ball_dof
        qvel = self.data.qvel
        rvx = float(qvel[start] - self.wind[0])
        rvy = float(qvel[start + 1] - self.wind[1])
        rvz = float(qvel[start + 2] - self.wind[2])
        speed_sq = rvx*rvx + rvy*rvy + rvz*rvz
        speed = float(np.sqrt(speed_sq))
        rotation = self.data.xmat[self.ball_body]
        local_ox, local_oy, local_oz = qvel[start+3:start+6]
        wx = float(rotation[0]*local_ox + rotation[1]*local_oy + rotation[2]*local_oz)
        wy = float(rotation[3]*local_ox + rotation[4]*local_oy + rotation[5]*local_oz)
        wz = float(rotation[6]*local_ox + rotation[7]*local_oy + rotation[8]*local_oz)
        cx, cy, cz = wy*rvz - wz*rvy, wz*rvx - wx*rvz, wx*rvy - wy*rvx
        cross_sq = cx*cx + cy*cy + cz*cz
        cross_length = float(np.sqrt(cross_sq))
        area = np.pi*b["radius_m"]**2
        dynamic = .5*b["air_density_kg_m3"]*area*speed_sq
        drag_factor = -.5*b["air_density_kg_m3"]*b["drag_coefficient"]*area*speed
        fx, fy, fz = drag_factor*rvx, drag_factor*rvy, drag_factor*rvz
        if speed > 1e-8 and cross_length > 1e-8:
            spin_ratio = b["radius_m"]*cross_length/speed**2
            lift = min(b.get("magnus_lift_max", .5), b.get("magnus_lift_slope", .15)*spin_ratio)
            lift_factor = dynamic*lift/cross_length
            fx, fy, fz = fx + lift_factor*cx, fy + lift_factor*cy, fz + lift_factor*cz
        applied = self.data.xfrc_applied[self.ball_body]
        applied[0], applied[1], applied[2] = fx, fy, fz
        # 空心薄壳球的转动惯量各向同性；下式在世界坐标系施加自旋阻尼力矩。
        decay = b.get("spin_decay_time_s", 2.)
        torque = -self.model.body_inertia[self.ball_body, 0]/decay
        applied[3], applied[4], applied[5] = torque*wx, torque*wy, torque*wz

    def _info(self):
        """汇总回合统计；其中球真值只用于离线评分和诊断，不进入观测。"""
        rmse = lambda samples: float(np.sqrt(np.mean(np.square(samples)))) if samples else None
        reached = self.bounces >= self.config["experiment"]["success_bounces"]
        rms = np.sqrt(self.torque_square_integral/max(float(self.data.time), 1e-12))
        return {"bounces": self.bounces, "success": reached and self.termination == "time_limit",
                "reached_target_bounces": reached,
                "apex_heights_m": list(self.apex_heights), "max_torque_ratio": self.max_torque_ratio,
                "max_joint_speed_rad_s": self.max_joint_speed, "termination": self.termination,
                "rms_torque_nm": rms.tolist(),
                "max_rms_rated_torque_ratio": float(np.max(rms/self.servo.rated_torque)),
                "horizontal_rmse_m": rmse(self.horizontal_errors),
                "bounce_radial_rmse_m": rmse(self.bounce_radial_errors),
                "max_bounce_radial_error_m": max(self.bounce_radial_errors, default=None),
                "sim_time_s": float(self.data.time), "observation_source": "stereo_rgb",
                "vision_position_rmse_m": rmse(self.position_errors), "vision_velocity_rmse_m_s": rmse(self.velocity_errors),
                "flight_position_rmse_m": rmse(self.flight_position_errors),
                "flight_velocity_rmse_m_s": rmse(self.flight_velocity_errors),
                "vision_valid_fraction": self.valid_vision_samples/max(1, self.vision_samples),
                "camera_frames": self.observer.frames_captured,
                "camera_missed_fraction": self.observer.frames_missed/max(1, self.observer.frames_captured),
                "mean_frame_age_s": float(np.mean(self.frame_ages)) if self.frame_ages else None}

    def step(self, action):
        """执行一个策略周期，并返回视觉观测、奖励、终止标志和诊断信息。"""
        # 一个策略步包含多个更细的物理步；碰撞和空气动力学都在物理频率更新。
        if self._done:
            raise RuntimeError("reset() before stepping a finished episode")
        a = np.asarray(action, dtype=float)
        if a.shape == (1,):
            a = np.array([0., 0., a[0], 0., 0.])
        if a.shape != (5,) or not np.isfinite(a).all():
            raise ValueError("residual action must have five finite values: XYZ velocity and two tilt rates")
        a = np.clip(a, -1., 1.)
        c, b = self.config["control"], self.config["ball"]
        residual_scale = np.r_[self._residual_velocity_scale(), self._residual_tilt_scale()]
        reward, terminated = 0., False
        for i in range(self.frame_skip):
            if i % self.servo_skip == 0:
                estimate = self.observer.observe(self.data)
                paddle, pv, _ = self.servo.paddle_state(self.data)
                # 在本地控制时钟更新规则基线，使新相机帧无需等待下一次策略调用。
                if estimate.valid:
                    command = self._camera_baseline(estimate, paddle, pv) + a*residual_scale
                else:
                    command = np.r_[np.clip(3*(self.servo.home_xyz[:2]-paddle[:2]), -.15, .15),
                                    np.clip(8*(self.baseline.home_z-paddle[2]), -.35, .35),
                                    -self.servo.cartesian_tilt/.08]
                self.servo.apply_cartesian(self.data, command[:3], command[3:], self.actuator_gain)
            vz_before = self.data.qvel[self.ball_dof+2]
            self.apply_ball_aerodynamics()
            mujoco.mj_step(self.model, self.data)
            # 以下分支只负责仿真计分和终止判定，不会污染策略观测。
            position, velocity = self.truth_ball_state()
            z, vz = position[2], velocity[2]
            paddle = self.data.site_xpos[self.site]
            rotation = self.data.site_xmat[self.site].reshape(3, 3)
            local = rotation.T @ (position-paddle)
            contact = self._contact()
            if contact and not self.in_contact:
                self.contact_started, self.contact_vin = self.data.time, vz_before
                self.bounce_radial_error = float(np.linalg.norm(local[:2]))
            if not contact and self.in_contact:
                if self.contact_vin < -.2 and vz > .3 and self.data.time-self.last_bounce_time > .1:
                    self.bounces += 1
                    self.last_bounce_time = self.data.time
                    self.bounce_radial_errors.append(self.bounce_radial_error)
                    reward += 1.
            self.in_contact = contact
            if vz <= 0 < self.last_apex_v and self.bounces:
                height = z-self.baseline.home_z-b["radius_m"]
                self.apex_heights.append(float(height))
                reward += 2*np.exp(-((height-self.config["scene"]["target_apex_m"])/.035)**2)
            self.last_apex_v = vz
            self.max_torque_ratio = max(self.max_torque_ratio, float(np.max(np.abs(self.data.ctrl)/self.model.actuator_ctrlrange[:, 1])))
            self.max_joint_speed = max(self.max_joint_speed, float(np.max(np.abs(self.data.qvel[:6]))))
            self.torque_square_integral += self.data.ctrl**2 * self.model.opt.timestep
            reason = ""
            radial = np.linalg.norm(local[:2])
            # 使用真实移动拍面坐标判定失球；没有隐藏挡板，也不会把球重置回来。
            if self.edge_contact:
                reason = "paddle_edge_missed"
            elif vz < 0 and local[2] <= b["radius_m"]+.003 and radial > self.config["paddle"]["radius_m"]+b["radius_m"] and not contact:
                reason = "paddle_missed"
            elif local[2] < -b["radius_m"] and not contact:
                reason = "ball_dropped"
            elif radial > .40:
                reason = "ball_side_escape"
            elif z > self.baseline.home_z + .65:
                reason = "ball_too_high"
            elif contact and self.data.time-self.contact_started > c["max_contact_s"]:
                reason = "sustained_contact"
            elif not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                reason = "nonfinite_state"
            elif np.any(self.data.qpos[:6] < self.model.jnt_range[:6, 0]-.01) or np.any(self.data.qpos[:6] > self.model.jnt_range[:6, 1]+.01):
                reason = "joint_limit"
            elif np.any(np.abs(paddle[:2] - self.servo.home_xyz[:2]) > self.servo.xy_tracking_limit+.025):
                reason = "paddle_workspace"
            if reason:
                terminated, self.termination = True, reason
                reward -= 3.
                break
        self.steps += 1
        # 横向居中项只是学习信号，不会对球施加额外力；这里的真值仅用于计分。
        self.horizontal_errors.append(float(radial))
        reward += (1. - .2*float(np.mean(a*a)) - .4*float(np.mean((a-self.last_action)**2))
                   - .4*min(float((radial/.05)**2), 4.)) / c["policy_hz"]
        self.last_action = a.astype(np.float32)
        observation = self._observation()
        self._record_vision_metrics()
        if not terminated and self.data.time-self.last_visual_valid_time > self.config["perception"]["tracking_timeout_s"]:
            terminated, self.termination = True, "vision_lost"
            reward -= 3.
        truncated = self.steps >= round(self.config["experiment"]["episode_s"] * c["policy_hz"])
        if truncated and not terminated:
            self.termination = "time_limit"
        self._done = terminated or truncated
        return observation, float(reward), terminated, truncated, self._info()

    def render(self):
        if self.renderer is None:
            camera = self.config["cameras"]["tracking"]
            self.renderer = mujoco.Renderer(self.model, camera["height"], camera["width"])
        self.renderer.update_scene(self.data, camera="pingpong_tracking")
        return self.renderer.render()

    def close(self):
        self.observer.close()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# 新代码使用的推荐名称；保留旧别名可兼容现有导入，不需要复制环境实现或旧模型。
PingPongEnv = PingPong1DEnv
