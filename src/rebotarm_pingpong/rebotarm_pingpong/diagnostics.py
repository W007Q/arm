"""乒乓球接触参数和相机视场的离线检查工具。

这里使用仿真真值来测量接触结果和几何覆盖范围，结果只用于校准和诊断，
不会作为控制器的观测输入。接触测试使用真正固定的拍面，不启动机械臂伺服，
也不创建图像渲染器，因此可以在无界面的环境中快速运行。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from rebotarm_description.parameters import description_path, joint_limits, load_hardware
from rebotarm_description.pingpong_model import derived_pingpong_files


def _model_sources(config_path=None):
    """读取统一配置，并生成诊断所需的乒乓球模型文件。

    模型仍通过和正式环境相同的参数入口构建，避免诊断脚本悄悄使用另一份
    关节限位、硬件尺寸或球拍参数。
    """
    root = description_path()
    config = yaml.safe_load(Path(config_path or root / "config/pingpong.yaml").read_text())
    files = derived_pingpong_files((root / "mjcf/robot.xml").read_text(), config,
                                   load_hardware(root / "config/rebotarm_hardware.yaml"), joint_limits(root))
    return root, config, files


def _drop_trial(config, files, height, timestep, *, xy=(0., 0.), horizontal_velocity=(0., 0.),
                spin=(0., 0., 0.), tilt=0., allow_miss=False):
    """让自由薄壳球落到固定拍面，返回碰撞和反弹的测量值。

    球的横向初速度、初始旋转和拍面倾角可单独设置，用来覆盖偏心碰撞、
    侧飞及旋转等情况。`allow_miss` 仅用于记录拍面外的失败样本。
    """
    robot = ET.fromstring(files["robot_pingpong.xml"])
    scene = ET.fromstring(files["pingpong_3d_scene.xml"])
    root = ET.Element("mujoco", model="fixed_paddle_contact_audit")
    option = copy.deepcopy(scene.find("option"))
    option.set("timestep", str(timestep))
    root.append(option)
    world = ET.SubElement(root, "worldbody")
    rot = Rotation.from_euler("y", tilt)
    support = ET.SubElement(world, "body", quat=" ".join(map(str, rot.as_quat()[[3, 0, 1, 2]])))
    support.append(copy.deepcopy(robot.find(".//geom[@name='paddle_face']")))
    ball = copy.deepcopy(scene.find(".//body[@name='pingpong_ball']"))
    contact_height = config["paddle"]["thickness_m"] / 2 + config["ball"]["radius_m"]
    ball.set("pos", f"{xy[0]} {xy[1]} {contact_height + height}")
    world.append(ball)
    root.append(copy.deepcopy(scene.find("contact")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qvel[:2] = horizontal_velocity
    data.qvel[3:6] = spin
    mujoco.mj_forward(model, data)
    normal = rot.as_matrix()[:, 2]
    b = config["ball"]
    mass = b["mass_kg"]
    inertia = model.body("pingpong_ball").inertia[0]
    gravity = config["scene"]["gravity_m_s2"]
    body_id = model.body("pingpong_ball").id
    area = np.pi * b["radius_m"] ** 2
    force = np.zeros(6)

    def energy():
        """计算球的平动、转动和重力势能，用于检查接触是否凭空增能。"""
        return float(.5*mass*np.dot(data.qvel[:3], data.qvel[:3])
                     + .5*inertia*np.dot(data.qvel[3:6], data.qvel[3:6])
                     + mass*gravity*(data.qpos[2]-contact_height))

    def advance():
        """施加空气阻力、马格努斯力和旋转衰减，然后推进一个时间步。

        这是固定拍面标定夹具，拍面不会主动运动，因而接触不能从机械臂命令
        获得额外能量。
        """
        # 固定拍面不会主动运动，接触不能从机械臂命令获得额外能量。
        velocity = data.qvel[:3]
        speed = np.linalg.norm(velocity)
        omega = data.xmat[body_id].reshape(3, 3) @ data.qvel[3:6]
        air_force = -.5*b["air_density_kg_m3"]*b["drag_coefficient"]*area*speed*velocity
        cross = np.cross(omega, velocity)
        cross_length = np.linalg.norm(cross)
        if speed > 1e-8 and cross_length > 1e-8:
            spin_ratio = b["radius_m"]*cross_length/speed**2
            lift = min(b["magnus_lift_max"], b["magnus_lift_slope"]*spin_ratio)
            air_force += .5*b["air_density_kg_m3"]*area*speed**2*lift*cross/cross_length
        data.xfrc_applied[body_id, :3] = air_force
        data.xfrc_applied[body_id, 3:] = -inertia*omega/b["spin_decay_time_s"]
        mujoco.mj_step(model, data)

    previous_contact = False
    incoming = outgoing = duration = None
    peak_penetration = peak_normal_force = normal_impulse = 0.
    for _ in range(round(.8/timestep)):
        before_velocity = data.qvel[:3].copy()
        before_energy = energy()
        advance()
        contact = data.ncon > 0
        for contact_id, pair in enumerate(data.contact):
            peak_penetration = max(peak_penetration, -float(pair.dist))
            mujoco.mj_contactForce(model, data, contact_id, force)
            peak_normal_force = max(peak_normal_force, float(force[0]))
            normal_impulse += max(0., float(force[0]))*timestep
        if contact and not previous_contact:
            incoming = float(np.dot(before_velocity, normal))
            start = data.time
            incoming_energy = before_energy
        if previous_contact and not contact:
            outgoing = float(np.dot(data.qvel[:3], normal))
            duration = data.time-start
            outgoing_energy = energy()
            outgoing_velocity = data.qvel[:3].copy()
            outgoing_spin = data.qvel[3:6].copy()
            break
        previous_contact = contact
    if incoming is None or outgoing is None:
        if allow_miss:
            return {"impact": False, "height_m": height, "initial_xy_m": list(xy),
                    "final_position_m": data.qpos[:3].tolist()}
        raise RuntimeError("Fixed-paddle drop audit produced no separated impact")
    apex = float(data.qpos[2])
    for _ in range(round(.6/timestep)):
        if data.qvel[2] <= 0:
            break
        advance()
        apex = max(apex, float(data.qpos[2]))
    return {"impact": True, "height_m": float(height), "timestep_s": float(timestep),
            "incoming_relative_m_s": incoming, "outgoing_relative_m_s": outgoing,
            "effective_restitution": float(outgoing/-incoming), "contact_s": float(duration),
            "rebound_height_m": max(0., apex-contact_height),
            "max_penetration_m": peak_penetration, "peak_normal_force_n": peak_normal_force,
            "normal_impulse_n_s": normal_impulse,
            "impact_energy_ratio": float(outgoing_energy/incoming_energy),
            "initial_xy_m": list(xy), "initial_spin_rad_s": list(spin), "paddle_tilt_rad": tilt,
            "outgoing_velocity_m_s": outgoing_velocity.tolist(), "outgoing_spin_rad_s": outgoing_spin.tolist()}


def _calibration_comparison(config, files):
    """将仿真落球结果与配置中的实测数据逐项比较。

    除反弹速度比外，还检查接触持续时间和反弹高度；至少覆盖三个落球高度
    才允许标记为通过，从而避免只用单个样本误判接触参数。
    """
    calibration = config["calibration"]
    samples = calibration["measurements"]
    comparisons = []
    for sample in samples:
        simulated = _drop_trial(config, files, float(sample["drop_height_m"]), config["scene"]["timestep_s"])
        measured_e = sample["outgoing_speed_m_s"]/sample["incoming_speed_m_s"]
        comparisons.append({"drop_height_m": sample["drop_height_m"],
            "measured_restitution": measured_e, "simulated_restitution": simulated["effective_restitution"],
            "restitution_abs_error": abs(simulated["effective_restitution"]-measured_e),
            "contact_duration_relative_error": abs(simulated["contact_s"]-sample["contact_duration_s"])/sample["contact_duration_s"],
            "rebound_height_relative_error": abs(simulated["rebound_height_m"]-sample["rebound_height_m"])/sample["rebound_height_m"]})
    e_tolerance = calibration["restitution_tolerance"]
    duration_tolerance = calibration["contact_duration_relative_tolerance"]
    rebound_tolerance = calibration.get("rebound_height_relative_tolerance", .10)
    height_coverage = len({round(float(t["drop_height_m"]), 4) for t in comparisons}) >= 3
    accepted = (calibration["status"] == "measured" and height_coverage
                and all(t["restitution_abs_error"] <= e_tolerance
                        and t["contact_duration_relative_error"] <= duration_tolerance
                        and t["rebound_height_relative_error"] <= rebound_tolerance for t in comparisons))
    return {"status": calibration["status"], "ball_id": calibration["ball_id"],
            "paddle_id": calibration["paddle_id"], "measurements_count": len(samples),
            "fit_within_tolerance": bool(accepted), "three_distinct_drop_heights": height_coverage,
            "comparisons": comparisons,
            "tolerances": {"restitution_abs": e_tolerance, "contact_duration_relative": duration_tolerance,
                           "rebound_height_relative": rebound_tolerance}}


def drop_report(*, config_path=None):
    """运行时间步收敛、偏心碰撞和拍面外落球测试并汇总结果。

    返回值是可序列化字典，供命令行诊断和文档中的校准报告直接使用。
    """
    _, config, files = _model_sources(config_path)
    dt = config["scene"]["timestep_s"]
    heights = config["calibration"]["drop_heights_m"]
    trials = [_drop_trial(config, files, height, dt*factor) for height in heights for factor in (1., .5, .25)]
    errors = [abs(trials[i]["effective_restitution"]-trials[i+2]["effective_restitution"])/trials[i+2]["effective_restitution"]
              for i in range(0, len(trials), 3)]
    stress = [_drop_trial(config, files, .15, dt, xy=(.005, -.004), horizontal_velocity=(.15, -.08)),
              _drop_trial(config, files, .15, dt, xy=(.040, .015), horizontal_velocity=(-.10, .04), spin=(30., -40., 20.)),
              _drop_trial(config, files, .15, dt, spin=(50., -30., 0.), tilt=.08)]
    miss = _drop_trial(config, files, .15, dt, xy=(config["paddle"]["radius_m"]+.05, 0.), allow_miss=True)
    comparison = _calibration_comparison(config, files)
    passive = all(t["impact_energy_ratio"] <= 1.01 for t in trials+stress)
    within_ten = max(errors) < .10
    return {"fixture": "free thin-shell ball and truly fixed paddle; no arm servo or renderer",
            "trials": trials, "stress_trials": stress, "off_paddle_trial": miss,
            "max_relative_restitution_change": max(errors), "within_10_percent": within_ten,
            "passive_energy_checks_passed": passive,
            "max_penetration_m": max(t["max_penetration_m"] for t in trials),
            "max_penetration_fraction_of_ball_radius": max(t["max_penetration_m"] for t in trials)/config["ball"]["radius_m"],
            "calibration_comparison": comparison,
            "physical_calibrated": bool(within_ten and passive and comparison["fit_within_tolerance"]),
            "note": "A passing report establishes numerical resolution/passivity. Physical agreement additionally requires measured rebound, speed and contact duration for this ball/rubber."}


def project(model, data, camera_id, point, width, height):
    """把世界坐标点投影到指定相机的像素坐标，并返回深度和焦距。

    该几何投影只服务于视场检查，不替代真实的 RGB 检测或双目三角测量。
    """
    p = data.cam_xmat[camera_id].reshape(3, 3).T @ (point-data.cam_xpos[camera_id])
    depth = -p[2]
    focal = .5*height/math.tan(math.radians(model.cam_fovy[camera_id])/2)
    return np.array([width/2+focal*p[0]/depth, height/2-focal*p[1]/depth]), depth, focal


def camera_report(*, config_path=None):
    """评估两台固定相机对颠球高度范围的覆盖和遮挡情况。

    报告包含球在画面中的比例、像素直径和在机械臂初始姿态下的遮挡率；
    动态遮挡、标定误差和图像延迟仍需通过视觉模块的专项测试测量。
    """
    from .control import horizontal_home
    root, config, files = _model_sources(config_path)
    assets = {p.name: p.read_bytes() for p in (root / "meshes").glob("*.stl")}
    assets.update({name: value.encode() for name, value in files.items()})
    model = mujoco.MjModel.from_xml_string(files["pingpong_3d_scene.xml"], assets)
    data = mujoco.MjData(model)
    data.qpos[:6] = horizontal_home(model, config)
    mujoco.mj_forward(model, data)
    xyz = np.array(config["scene"]["home_surface_xyz_m"])
    zmin = xyz[2]-config["control"]["stroke_m"]+config["ball"]["radius_m"]
    zmax = xyz[2]+config["ball"]["radius_m"]+2*config["scene"]["target_apex_m"]
    ball_body = model.body("pingpong_ball").id
    cameras = {}
    width, height = config["perception"]["width"], config["perception"]["height"]
    for name in config["perception"]["camera_names"]:
        camera = config["cameras"][name]
        cid = model.camera(f"pingpong_{name}").id
        samples = []
        geomid = np.zeros(1, dtype=np.int32)
        for z in np.linspace(zmin, zmax, 31):
            point = xyz.copy()
            point[2] = z
            pixel, depth, focal = project(model, data, cid, point, width, height)
            direction = point-data.cam_xpos[cid]
            distance = np.linalg.norm(direction)
            direction /= distance
            hit = mujoco.mj_ray(model, data, data.cam_xpos[cid], direction, None, 1, ball_body, geomid)
            occluded = hit >= 0 and hit < distance-config["ball"]["radius_m"]
            in_frame = depth > 0 and 0 <= pixel[0] < width and 0 <= pixel[1] < height
            samples.append({"z_m": float(z), "u_px": float(pixel[0]), "v_px": float(pixel[1]),
                            "ball_diameter_px": float(2*focal*config["ball"]["radius_m"]/depth),
                            "in_frame": bool(in_frame), "occluded_at_home": bool(occluded)})
        cameras[name] = {"position_m": camera["position_m"], "look_at_m": camera["look_at_m"],
                        "resolution": [width, height], "samples": samples,
                        "in_frame_fraction": float(np.mean([s["in_frame"] for s in samples])),
                        "unoccluded_at_home_fraction": float(np.mean([not s["occluded_at_home"] for s in samples])),
                        "min_ball_diameter_px": min(s["ball_diameter_px"] for s in samples)}
    velocity = math.sqrt(2*config["scene"]["gravity_m_s2"]*config["scene"]["target_apex_m"])
    return {"mount": "two fixed external cameras with calibrated world/base poses",
            "observation_source": "rendered stereo RGB centroids, triangulation and timestamped filter",
            "cameras": cameras, "resolution": [width, height], "fps": config["perception"]["fps"],
            "trajectory_height_range_m": [zmin, zmax],
            "in_frame_fraction": min(c["in_frame_fraction"] for c in cameras.values()),
            "unoccluded_at_home_fraction": min(c["unoccluded_at_home_fraction"] for c in cameras.values()),
            "min_ball_diameter_px": min(c["min_ball_diameter_px"] for c in cameras.values()),
            "near_impact_speed_m_s": velocity,
            "travel_per_frame_mm": {str(fps): velocity/fps*1000 for fps in (30, 60, 90, 120)},
            "note": "Geometry is sampled at the arm home pose only. It does not verify RGB detection, calibration errors, moving-arm occlusion or image latency; those are measured separately."}
