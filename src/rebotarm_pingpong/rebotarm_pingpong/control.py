"""乒乓球控制器：垂直基线控制和受力矩限制的六轴笛卡尔伺服。

控制器只向 MuJoCo 发送电机目标（位置、速度和前馈力矩），不直接修改
球或机械臂状态。这样仿真中的碰撞、延迟和电机限幅会自然影响结果，便于
后续把相同的控制接口迁移到真机。
"""
from __future__ import annotations

from collections import deque
import copy

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from rebotarm_description.parameters import ARM_JOINTS, arm_gains, hardware_profile, joint_limits, load_hardware, uart_parameters


def horizontal_home(model, config):
    """在启动时求解拍面水平零位；不会覆盖正在运行的仿真状态。"""
    data = mujoco.MjData(model)
    site = model.site("paddle_surface").id
    target = np.array(config["scene"]["home_surface_xyz_m"])
    target_rotation = Rotation.from_euler("z", config["scene"]["home_yaw_rad"]).as_matrix()
    lo, hi = model.jnt_range[:6, 0] + .03, model.jnt_range[:6, 1] - .03

    def error(q):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        return np.r_[data.site_xpos[site] - target,
                     .2 * Rotation.from_matrix(data.site_xmat[site].reshape(3, 3) @ target_rotation.T).as_rotvec()]

    rng = np.random.default_rng(0)
    guesses = [np.array([-.1890, -1.0372, 1.4, -.7829, -.4, -1.8])]
    guesses.extend(rng.uniform(lo, hi, size=(8, 6)))
    for q in guesses:
        result = least_squares(error, np.clip(q, lo, hi), bounds=(lo, hi), max_nfev=180)
        if np.linalg.norm(result.fun) < 1e-6:
            return result.x
    raise ValueError("Horizontal paddle home is unreachable; adjust mount/home in pingpong.yaml")


class VerticalBaseline:
    """一维基线控制：根据碰撞前后的能量选择拍面竖直速度。

    该类只负责旧版的一维任务。自由落球任务使用 ``CartesianServo`` 的
    三维接口，并由相机估计提供球的位置和速度。
    """

    def __init__(self, config):
        self.config = config
        self.home_z = config["scene"]["home_surface_xyz_m"][2]
        self.gravity = config["scene"]["gravity_m_s2"]
        self.radius = config["ball"]["radius_m"]

    def command(self, ball_z, ball_vz, paddle_z, paddle_vz):
        """根据球和拍面的竖直状态返回拍面速度（单位：m/s）。"""
        c = self.config["control"]
        gap = max(0., ball_z - paddle_z - self.radius)
        # 这里先按静止拍面预测；PPO 残差再补偿碰撞损失、伺服滞后和观测误差。
        impact_v = -np.sqrt(max(0., ball_vz**2 + 2 * self.gravity * gap))
        impact_t = (ball_vz - impact_v) / self.gravity
        apex_world = self.home_z + self.radius + self.config["scene"]["target_apex_m"]
        desired_v = np.sqrt(2 * self.gravity * max(.03, apex_world - paddle_z - self.radius))
        e = c["nominal_restitution"]
        hit_speed = (desired_v + e * impact_v) / (1 + e)
        if ball_vz < 0 and impact_t < .09:
            return float(np.clip(hit_speed, -.2, c["max_paddle_speed_m_s"]))
        # 球在空中飞行时，让拍面回到行程中点，准备下一次击球。
        return float(np.clip(8 * (self.home_z - paddle_z), -.35, .35))


class CartesianServo:
    """基于编码器的标称控制，包含通信延迟和有限响应的电机模型。

    目标是关节位置/速度，真正进入仿真 plant 的只有受限电机力矩。不使用
    mocap 拍子、球体冲量、位置重置或 weld。延迟、滞后和线性速度/力矩包络
    是待实测的先验参数，不代表已完成的驱动器辨识。
    """

    def __init__(self, model, config):
        """读取硬件约束并预计算竖直轨迹及三维控制所需的缓存。"""
        self.model, self.config = model, config
        self.site = model.site("paddle_surface").id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.mass = np.zeros((model.nv, model.nv))
        # 标称控制模型与随机化、碰撞和球状态隔离；这里只复制六个机械臂关节
        # 编码器，标称模型中的自由球位置和速度始终保持默认值。
        self.nominal_model = copy.copy(model)
        self.nominal_model.geom_contype[:] = 0
        self.nominal_model.geom_conaffinity[:] = 0
        self.nominal_data = mujoco.MjData(self.nominal_model)
        self.nominal_jacp = np.zeros_like(self.jacp)
        self.nominal_jacr = np.zeros_like(self.jacr)
        # reference 是拍面目标位置；velocity 是旧一维接口的竖直目标速度。
        self.reference = np.array(config["scene"]["home_surface_xyz_m"], dtype=float)
        self.velocity = 0.
        hardware = load_hardware()
        gains = arm_gains(hardware)
        self.kp = np.array(gains["kp"]) * config["control"]["joint_kp_scale"]
        self.kd = np.array(gains["kd"]) * config["control"]["joint_kd_scale"]
        joints = {j["name"]: j for j in hardware_profile(hardware)["joints"]}
        motors = [hardware["motor_catalog"][joints[name]["model"]] for name in ARM_JOINTS]
        # 电机能力：空载转速、关节速度、峰值/额定力矩。
        self.no_load_speed = np.array([float(motor["no_load_speed"]) for motor in motors])
        limits = joint_limits()
        transport = uart_parameters(hardware, limits)
        transport_speed = np.asarray(transport["velocity_limit"][:6])
        self.transport_hz = float(transport["tx_hz"])
        if not np.isfinite(self.transport_hz) or self.transport_hz <= 0:
            raise ValueError("UART transport tx_hz must be finite and positive")
        # 电机本地反馈和主机通信使用不同的时钟。即使 apply 以 500 Hz 调用，
        # 当前 50 Hz 的 UART 链路也不会因此发送更多目标。
        # 主机发送频率不能超过 UART；本地伺服频率可以更高。
        self.command_hz = min(self.transport_hz, float(config["control"]["servo_hz"]))
        self.command_period_s = 1. / self.command_hz
        # 与真机共用逐轴限速，包括 STM32 固件按槽位设置的上限。
        self.joint_speed_limit = np.minimum.reduce([
            self.no_load_speed,
            np.asarray([limits[name]["velocity"] for name in ARM_JOINTS]),
            transport_speed,
        ])
        self.peak_torque = np.minimum(np.max(np.abs(model.actuator_ctrlrange), axis=1),
                                      [float(motor["peak"]) for motor in motors])
        self.rated_torque = np.array([float(motor["rated"]) for motor in motors])
        self.actuator_delay_s = float(config["control"].get("actuator_delay_s", .004))
        self.actuator_time_constant_s = float(config["control"].get("actuator_time_constant_s", .004))
        if not np.isfinite(self.actuator_delay_s) or self.actuator_delay_s < 0:
            raise ValueError("actuator_delay_s must be finite and nonnegative")
        if not np.isfinite(self.actuator_time_constant_s) or self.actuator_time_constant_s <= 0:
            raise ValueError("actuator_time_constant_s must be finite and positive")
        if np.any(self.no_load_speed <= 0) or np.any(self.joint_speed_limit <= 0):
            raise ValueError("motor speed capabilities must be positive")
        # 启动时一次性求解带边界的旧版竖直轨迹。三次样条直接给出曲率导数，
        # 使用该接口时每周期无需对接近奇异的雅可比做非线性求逆。
        data = mujoco.MjData(model)
        target_rotation = Rotation.from_euler("z", config["scene"]["home_yaw_rad"]).as_matrix()
        home = horizontal_home(model, config)
        lo, hi = model.jnt_range[:6, 0] + .02, model.jnt_range[:6, 1] - .02
        stroke = config["control"]["stroke_m"]
        heights = np.linspace(self.reference[2]-stroke, self.reference[2]+stroke, 61)
        solutions = []
        q = home.copy()
        for z in heights:
            target = self.reference.copy()
            target[2] = z
            def error(q):
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)
                return np.r_[data.site_xpos[self.site]-target,
                             .2*Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3) @ target_rotation.T).as_rotvec()]
            result = least_squares(error, np.clip(q, lo, hi), bounds=(lo, hi), max_nfev=100)
            if np.linalg.norm(result.fun) > 1e-5:
                raise ValueError("Vertical paddle stroke is unreachable; reduce stroke_m or adjust mount/home")
            q = result.x
            solutions.append(q)
        self.path = CubicSpline(heights, np.array(solutions))
        # 旧任务只控制竖直坐标，因此保留这条轨迹；自由球任务另外暴露有界
        # 笛卡尔目标。以下限值是保守先验，真机使用前仍应按工作空间和电机
        # 实测结果校准。
        control = config["control"]
        self.home_xyz = self.reference.copy()
        self.home_rotation = target_rotation.copy()
        self.xy_tracking_limit = float(control.get("xy_tracking_limit_m", .08))
        if not np.isfinite(self.xy_tracking_limit) or self.xy_tracking_limit <= 0:
            raise ValueError("xy_tracking_limit_m must be finite and positive")
        # 三维速度/加速度和拍面倾角限幅，防止追踪噪声造成不现实的动作。
        self.cartesian_speed_limit = np.asarray(
            control.get("cartesian_speed_limit_m_s", [.25, .25, control["max_paddle_speed_m_s"]]),
            dtype=float)
        self.cartesian_acceleration_limit = np.asarray(
            control.get("cartesian_acceleration_limit_m_s2", [.8, .8, control["max_paddle_acceleration_m_s2"]]),
            dtype=float)
        if self.cartesian_speed_limit.shape != (3,) or self.cartesian_acceleration_limit.shape != (3,):
            raise ValueError("Cartesian speed and acceleration limits must have three entries")
        if (not np.isfinite(self.cartesian_speed_limit).all() or
                not np.isfinite(self.cartesian_acceleration_limit).all() or
                np.any(self.cartesian_speed_limit <= 0) or np.any(self.cartesian_acceleration_limit <= 0)):
            raise ValueError("Cartesian speed and acceleration limits must be finite and positive")
        self.tilt_limit_rad = float(control.get("paddle_tilt_limit_rad", .25))
        self.tilt_rate_limit = float(control.get("paddle_tilt_rate_limit_rad_s", 1.5))
        self.tilt_acceleration_limit = float(control.get("paddle_tilt_acceleration_limit_rad_s2", 8.))
        if (not np.isfinite(self.tilt_limit_rad) or self.tilt_limit_rad <= 0 or
                not np.isfinite(self.tilt_rate_limit) or self.tilt_rate_limit <= 0 or
                not np.isfinite(self.tilt_acceleration_limit) or self.tilt_acceleration_limit <= 0):
            raise ValueError("paddle tilt limits must be finite and positive")
        self._joint_lo = model.jnt_range[:6, 0] + .02
        self._joint_hi = model.jnt_range[:6, 1] - .02
        self.joint_target_acceleration_limit = float(control.get("joint_target_acceleration_rad_s2", 10.))
        if not np.isfinite(self.joint_target_acceleration_limit) or self.joint_target_acceleration_limit <= 0:
            raise ValueError("joint_target_acceleration_rad_s2 must be finite and positive")
        self._ik_target_q = home.copy()
        self._ik_target_velocity = np.zeros(6)
        self.cartesian_velocity = np.zeros(3)
        self.cartesian_tilt = np.zeros(2)
        self.cartesian_tilt_velocity = np.zeros(2)
        self.reset()

    def reset(self):
        """清空目标、延迟队列和电机状态，回到拍面 home 姿态。"""
        self.reference[:] = self.config["scene"]["home_surface_xyz_m"]
        self.velocity = 0.
        self.cartesian_velocity[:] = 0.
        self.cartesian_tilt[:] = 0.
        self.cartesian_tilt_velocity[:] = 0.
        # IK 始终从标称 home 解开始，不会使用球的位置或 mocap 对象作为初值。
        self._ik_target_q[:] = self.path(self.reference[2])
        self._ik_target_velocity[:] = 0.
        self._last_ik_command_q = self._ik_target_q.copy()
        self.pending_commands = deque()
        self.delayed_position = self.delayed_velocity = None
        self.delayed_feedforward = np.zeros(6)
        self.applied_torque = np.zeros(6)
        self.last_apply_time = None
        self.next_tx_time = 0.
        self.transmitted_commands = 0
        mujoco.mj_resetData(self.nominal_model, self.nominal_data)

    def _encoder_state(self, data):
        """把六个关节编码器读数复制到标称模型并执行正运动学。"""
        self.nominal_data.qpos[:6] = data.qpos[:6]
        self.nominal_data.qvel[:6] = data.qvel[:6]
        mujoco.mj_forward(self.nominal_model, self.nominal_data)

    def paddle_state(self, data):
        """返回由编码器正运动学得到的拍面位置、速度和旋转矩阵。

        球的位置/速度、接触状态和 plant 的 site 真值不会作为控制器输入。
        """
        self._encoder_state(data)
        mujoco.mj_jacSite(self.nominal_model, self.nominal_data,
                         self.nominal_jacp, self.nominal_jacr, self.site)
        position = self.nominal_data.site_xpos[self.site].copy()
        velocity = self.nominal_jacp[:, :6] @ self.nominal_data.qvel[:6]
        rotation = self.nominal_data.site_xmat[self.site].reshape(3, 3).copy()
        return position, velocity, rotation

    def target_from_ball_estimate(self, estimate, *, lookahead_s=.06,
                                  target_z=None, tilt_gain=None):
        """把相机估计转换为受限的拍面追踪目标。

        ``estimate`` 只需提供 ``position``、``velocity`` 和 ``valid`` 三个
        字段，因此无需依赖感知模块的具体类型。跟踪失效或过期时返回
        ``None``；调用方应回到 home，不能用仿真真值替代视觉结果。

        预测采用短时恒加速度外推。XY 目标限制在 home 附近的工作空间内；
        小幅滚转/俯仰修正抵消横向速度，硬倾角上限避免机械臂追逐噪声像素。
        """
        valid = bool(getattr(estimate, "valid", False))
        if not valid:
            return None
        position = np.asarray(getattr(estimate, "position", []), dtype=float)
        velocity = np.asarray(getattr(estimate, "velocity", []), dtype=float)
        if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
            return None
        horizon = float(lookahead_s)
        if not np.isfinite(horizon) or horizon < 0 or horizon > .30:
            raise ValueError("lookahead_s must be finite and in [0, .30]")
        gravity = float(self.config["scene"]["gravity_m_s2"])
        predicted = position + velocity * horizon
        predicted[2] -= .5 * gravity * horizon * horizon
        target = self.home_xyz.copy()
        target[:2] = self.home_xyz[:2] + np.clip(predicted[:2] - self.home_xyz[:2],
                                                  -self.xy_tracking_limit, self.xy_tracking_limit)
        target[2] = self.reference[2] if target_z is None else float(target_z)
        target[2] = np.clip(target[2], self.home_xyz[2] - self.config["control"]["stroke_m"],
                            self.home_xyz[2] + self.config["control"]["stroke_m"])
        # tilt[0] 是拍面局部 x 轴的横滚，tilt[1] 是局部 y 轴的俯仰。目标法向
        # 同时抵消横向速度和偏离 home 的位移，使球回到击球区域，而不是一直
        # 追到工作空间边缘。这里的增益是控制先验，并非碰撞标定参数。
        c = self.config["control"]
        velocity_gain = float(c.get("lateral_velocity_gain_rad_s_m", .10) if tilt_gain is None else tilt_gain)
        position_gain = float(c.get("lateral_position_gain_rad_m", .50))
        if not np.isfinite(velocity_gain) or not np.isfinite(position_gain) or min(velocity_gain, position_gain) < 0:
            raise ValueError("lateral tilt gains must be finite and nonnegative")
        desired_world_normal = -(velocity_gain*velocity[:2] + position_gain*(position[:2]-self.home_xyz[:2]))
        local_normal = self.home_rotation.T @ np.r_[desired_world_normal, 0.]
        tilt = np.clip(np.array([-local_normal[1], local_normal[0]]), -self.tilt_limit_rad, self.tilt_limit_rad)
        return target, tilt

    def _target_rotation(self):
        """根据当前局部滚转/俯仰目标计算拍面在世界坐标中的旋转。"""
        local_tilt = Rotation.from_euler("xy", self.cartesian_tilt).as_matrix()
        return self.home_rotation @ local_tilt

    def _damped_ik(self, target_xyz, target_rotation):
        """从编码器正运动学求解一小步、带边界的笛卡尔 IK 修正。

        使用阻尼雅可比而非每周期非线性优化。以上一条命令为初值，并限制
        每次关节步长，保证相机跳变时仍然因果且不越过关节限位。碰撞和球体
        状态不会参与该计算。
        """
        q = np.clip(self._ik_target_q.copy(), self._joint_lo, self._joint_hi)
        target_xyz = np.asarray(target_xyz, dtype=float)
        regularization = 2.5e-4
        for _ in range(8):
            self.nominal_data.qpos[:6] = q
            self.nominal_data.qvel[:6] = 0.
            mujoco.mj_forward(self.nominal_model, self.nominal_data)
            mujoco.mj_jacSite(self.nominal_model, self.nominal_data,
                              self.nominal_jacp, self.nominal_jacr, self.site)
            current = self.nominal_data.site_xpos[self.site]
            rotation = self.nominal_data.site_xmat[self.site].reshape(3, 3)
            position_error = target_xyz - current
            rotation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            error = np.r_[position_error, .20 * rotation_error]
            if np.linalg.norm(error) < 2e-5:
                break
            jacobian = np.vstack((self.nominal_jacp[:, :6], .20 * self.nominal_jacr[:, :6]))
            gram = jacobian @ jacobian.T + regularization * np.eye(6)
            try:
                step = jacobian.T @ np.linalg.solve(gram, error)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(jacobian, rcond=1e-4) @ error
            q = np.clip(q + np.clip(step, -.035, .035), self._joint_lo, self._joint_hi)
        self._ik_target_q[:] = q
        return q

    def apply_cartesian(self, data, requested_velocity_xyz, requested_tilt_rate_xy=(0., 0.), actuator_gain=1.):
        """执行受限的 XYZ 平移和拍面倾角速度命令。

        自由落球策略使用五维命令：三个笛卡尔速度和两个局部倾角速度。
        位置、加速度、工作空间、关节速度、力矩、传输延迟和电机滞后限幅
        始终有效。参数是命令而非瞬移，所有运动都由机械臂执行器产生。

        旧版检查点和诊断仍可使用 :meth:`apply` 的竖直接口；新策略应调用
        本方法，并且球的相关输入全部来自相机估计。
        """
        command = np.asarray(requested_velocity_xyz, dtype=float)
        tilt_command = np.asarray(requested_tilt_rate_xy, dtype=float)
        if command.shape != (3,) or tilt_command.shape != (2,) or not np.isfinite(command).all() or not np.isfinite(tilt_command).all():
            raise ValueError("Cartesian command must contain finite (3,) velocity and (2,) tilt-rate arrays")
        m, c = self.model, self.config["control"]
        now = float(data.time)
        dt = 1. / float(c["servo_hz"]) if self.last_apply_time is None else max(0., now - self.last_apply_time)
        self.last_apply_time = now
        self._encoder_state(data)
        command = np.clip(command, -self.cartesian_speed_limit, self.cartesian_speed_limit)
        self.cartesian_velocity += np.clip(command - self.cartesian_velocity,
                                           -self.cartesian_acceleration_limit * dt,
                                           self.cartesian_acceleration_limit * dt)
        self.cartesian_velocity = np.clip(self.cartesian_velocity, -self.cartesian_speed_limit, self.cartesian_speed_limit)
        self.reference += self.cartesian_velocity * dt
        bounds = np.array([self.xy_tracking_limit, self.xy_tracking_limit, c["stroke_m"]])
        self.reference = np.clip(self.reference, self.home_xyz - bounds, self.home_xyz + bounds)
        for axis in range(3):
            lower, upper = self.home_xyz[axis] - bounds[axis], self.home_xyz[axis] + bounds[axis]
            if ((self.reference[axis] <= lower + 1e-12 and self.cartesian_velocity[axis] < 0) or
                    (self.reference[axis] >= upper - 1e-12 and self.cartesian_velocity[axis] > 0)):
                self.cartesian_velocity[axis] = 0.
        tilt_command = np.clip(tilt_command, -self.tilt_rate_limit, self.tilt_rate_limit)
        self.cartesian_tilt_velocity += np.clip(tilt_command - self.cartesian_tilt_velocity,
                                                -self.tilt_acceleration_limit * dt,
                                                self.tilt_acceleration_limit * dt)
        self.cartesian_tilt_velocity = np.clip(self.cartesian_tilt_velocity, -self.tilt_rate_limit, self.tilt_rate_limit)
        self.cartesian_tilt += self.cartesian_tilt_velocity * dt
        self.cartesian_tilt = np.clip(self.cartesian_tilt, -self.tilt_limit_rad, self.tilt_limit_rad)
        for axis in range(2):
            if abs(self.cartesian_tilt[axis]) >= self.tilt_limit_rad - 1e-12:
                self.cartesian_tilt_velocity[axis] = 0.

        previous_target_q = self._last_ik_command_q.copy()
        unconstrained_target_q = self._damped_ik(self.reference, self._target_rotation())
        # 位置目标也必须遵守关节速度限值；只裁剪速度通道会让 MIT 位置环
        # 仍收到任意大的位置跳变。
        previous_dq = self._ik_target_velocity.copy()
        requested_dq = np.clip((unconstrained_target_q-previous_target_q)/max(dt, 1e-12),
                               -self.joint_speed_limit, self.joint_speed_limit)
        target_dq = previous_dq + np.clip(requested_dq-previous_dq,
                                          -self.joint_target_acceleration_limit*dt,
                                          self.joint_target_acceleration_limit*dt)
        target_q = np.clip(previous_target_q + target_dq*dt, self._joint_lo, self._joint_hi)
        target_dq = (target_q-previous_target_q)/max(dt, 1e-12)
        self._ik_target_q[:] = target_q
        # 抗积分饱和只作用于命令参考，不会改写仿真本体。如果工作空间、奇异
        # 姿态或电机限速使 IK 落后，位置目标最多保留 3 mm 欠量；关节目标还
        # 受速度和加速度限幅，避免不可达的长行程让电机持续饱和。
        self.nominal_data.qpos[:6] = target_q
        self.nominal_data.qvel[:6] = target_dq
        mujoco.mj_forward(self.nominal_model, self.nominal_data)
        reachable_position = self.nominal_data.site_xpos[self.site].copy()
        self.reference[:] = reachable_position + np.clip(self.reference-reachable_position, -.003, .003)
        self.reference[:] = np.clip(self.reference, self.home_xyz-bounds, self.home_xyz+bounds)
        # 关节轨迹受限追赶目标时，命令中的倾角状态仍独立保存。实际法向只作
        # 诊断反馈；把它再积分回命令会把暂时的 IK 滞后误认为倾角，放大横向
        # 冲量。IK 迭代会把 nominal_data 留在目标姿态，因此计算前馈和 PD 前
        # 要恢复编码器实测姿态，否则误差项会每次都被错误地算成零。
        self._encoder_state(data)
        # _damped_ik 保存最新目标；用上一条已发送目标估计 qdot，避免通信停滞
        # 时产生虚假的速度尖峰。
        self._ik_target_velocity[:] = target_dq
        self._last_ik_command_q = target_q.copy()
        qacc = (target_dq - previous_dq) / max(dt, 1e-12)
        mujoco.mj_fullM(self.nominal_model, self.nominal_data, self.mass)
        feedforward = self.mass[:6, :6] @ qacc + self.nominal_data.qfrc_bias[:6]
        self._apply_joint_target(data, now, dt, target_q, target_dq, feedforward, actuator_gain)

    def _apply_joint_target(self, data, now, dt, target_q, target_dq, feedforward, actuator_gain):
        """排队一条 MIT 目标，并应用延迟后的电机响应。"""
        m = self.model
        if now + 1e-10 >= self.next_tx_time:
            self.pending_commands.append((now + self.actuator_delay_s, target_q.copy(), target_dq.copy(), feedforward.copy()))
            self.transmitted_commands += 1
            skipped = int(max(0., now - self.next_tx_time + 1e-10) / self.command_period_s)
            self.next_tx_time += (skipped + 1) * self.command_period_s
        while self.pending_commands and self.pending_commands[0][0] <= now + 1e-12:
            _, self.delayed_position, self.delayed_velocity, self.delayed_feedforward = self.pending_commands.popleft()
        torque = np.zeros(6) if self.delayed_position is None else (
            self.kp*(self.delayed_position-self.nominal_data.qpos[:6])
            + self.kd*(self.delayed_velocity-self.nominal_data.qvel[:6]) + self.delayed_feedforward)
        torque = np.clip(torque, -self.peak_torque, self.peak_torque)
        alpha = -np.expm1(-dt / self.actuator_time_constant_s)
        self.applied_torque += alpha * (torque * actuator_gain - self.applied_torque)
        self.applied_torque = self._motor_output(self.applied_torque, self.nominal_data.qvel[:6])
        data.ctrl[:] = np.clip(self.applied_torque, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

    def _motor_output(self, requested_torque, joint_velocity):
        """近似反电动势造成的输出力矩包络，不重置关节速度。

        驱动方向的力矩随转速线性下降，到目录空载转速时为零；制动方向仍
        受峰值力矩限制。该近似假设驱动器支持制动，仍需用电压/电流实测。
        """
        available = self.peak_torque * np.clip(1 - np.abs(joint_velocity) / self.no_load_speed, 0., 1.)
        upper = np.where(joint_velocity > 0, available, self.peak_torque)
        lower = -np.where(joint_velocity < 0, available, self.peak_torque)
        return np.clip(requested_torque, lower, upper)

    def apply(self, data, requested_speed, actuator_gain=1.):
        """兼容旧版的一维竖直控制接口，输入为拍面目标速度（m/s）。"""
        m, c = self.model, self.config["control"]
        dt = 1 / c["servo_hz"]
        now = float(data.time)
        if self.last_apply_time is not None:
            dt = max(0., now - self.last_apply_time)
        self.last_apply_time = now
        self._encoder_state(data)
        home = self.config["scene"]["home_surface_xyz_m"][2]
        previous_velocity = self.velocity
        # 轨迹同时遵守主机通信的速度上限，不向真实链路请求它无法发送的 qdot。
        derivative = np.abs(self.path(self.reference[2], 1))
        vertical_limit = min(c["max_paddle_speed_m_s"],
                             float(np.min(self.joint_speed_limit / np.maximum(derivative, 1e-9))))
        requested_speed = float(np.clip(requested_speed, -vertical_limit, vertical_limit))
        self.velocity += np.clip(requested_speed - self.velocity,
                                 -c["max_paddle_acceleration_m_s2"] * dt,
                                 c["max_paddle_acceleration_m_s2"] * dt)
        self.velocity = float(np.clip(self.velocity, -vertical_limit, vertical_limit))
        self.reference[2] = np.clip(self.reference[2] + self.velocity * dt, home - c["stroke_m"], home + c["stroke_m"])
        if abs(self.reference[2] - home) >= c["stroke_m"] - 1e-9:
            self.velocity = 0.
        z = self.reference[2]
        target_q = self.path(z)
        target_dq = self.path(z, 1) * self.velocity
        qacc = self.path(z, 1) * ((self.velocity-previous_velocity)/max(dt, 1e-12)) + self.path(z, 2) * self.velocity**2
        mujoco.mj_fullM(self.nominal_model, self.nominal_data, self.mass)
        feedforward = self.mass[:6, :6] @ qacc + self.nominal_data.qfrc_bias[:6]
        # 硬件使用 MIT 模式：CAN/UART 传输位置、速度和前馈量，驱动器在本地
        # 根据编码器闭合 PD。若把这路反馈再当作主机力矩环延迟，会变成另一种
        # 且常常不稳定的控制架构。
        if now + 1e-10 >= self.next_tx_time:
            self.pending_commands.append((now + self.actuator_delay_s, target_q.copy(), target_dq.copy(), feedforward.copy()))
            self.transmitted_commands += 1
            skipped = int(max(0., now - self.next_tx_time + 1e-10) / self.command_period_s)
            self.next_tx_time += (skipped + 1) * self.command_period_s
        while self.pending_commands and self.pending_commands[0][0] <= now + 1e-12:
            _, self.delayed_position, self.delayed_velocity, self.delayed_feedforward = self.pending_commands.popleft()
        torque = np.zeros(6) if self.delayed_position is None else (
            self.kp*(self.delayed_position-self.nominal_data.qpos[:6])
            + self.kd*(self.delayed_velocity-self.nominal_data.qvel[:6]) + self.delayed_feedforward)
        torque = np.clip(torque, -self.peak_torque, self.peak_torque)
        alpha = -np.expm1(-dt / self.actuator_time_constant_s)
        self.applied_torque += alpha * (torque * actuator_gain - self.applied_torque)
        self.applied_torque = self._motor_output(self.applied_torque, self.nominal_data.qvel[:6])
        data.ctrl[:] = np.clip(self.applied_torque, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
