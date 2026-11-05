"""集中读取机器人模型、硬件参数和传感器安装位姿。

关节位置/速度/力矩上限以及惯量以 URDF 为准，电机能力、MIT 增益和通信
安全上限以 ``rebotarm_hardware.yaml`` 为准。相机外壳安装以 URDF 为准，
传感器外参和 TCP 以 ``d435_mount.yaml`` 为准。其它模块读取这些唯一来源，避免在
控制器、仿真和串口代码中重复维护同一组数字。这里仅读取配置，不会打开
硬件连接。
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))


def description_path() -> Path:
    """返回描述包的根目录，优先使用当前源码树。

    在源码目录运行同步脚本或测试时，优先读取相邻的 URDF/config，避免误
    用旧的安装副本；安装后则通过 ament 索引定位 share 目录。
    """
    # 源码树优先，避免同步脚本读到过期的安装文件。
    source = Path(__file__).resolve().parents[1]
    if (source / "urdf/robot_arm.urdf").is_file():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("rebotarm_description"))


def load_hardware(path: str | Path | None = None) -> dict:
    """读取电机目录、默认型号和 UART 安全策略。"""
    path = Path(path) if path else description_path() / "config/rebotarm_hardware.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_mounts(root: Path | None = None) -> dict:
    """读取 TCP 与 D435 相机相对于末端的安装参数。"""
    return yaml.safe_load(((root or description_path()) / "config/d435_mount.yaml").read_text(encoding="utf-8"))


def hardware_profile(hardware: dict) -> dict:
    """从硬件配置中取出默认型号的覆盖项。"""
    return hardware["models"][hardware["default_model"]]["overrides"]


def arm_gains(hardware: dict | None = None) -> dict[str, list[float]]:
    """按 ``joint1`` 到 ``joint6`` 顺序返回 MIT 的 kp/kd 增益。"""
    profile = hardware_profile(hardware or load_hardware())
    joints = {j["name"]: j for j in profile["joints"]}
    return {key: [float(joints[name]["MIT"][key]) for name in ARM_JOINTS] for key in ("kp", "kd")}


def uart_gains(hardware: dict | None = None) -> dict[str, list[float]]:
    """从每个逻辑关节的 MIT 字段读取 UART 增益。

    增益只在 ``models.*.overrides.joints[].MIT`` 中维护。UART 编码时
    再按 STM32 槽位映射重排，因此不需要另一份 default_kp/default_kd。
    """
    hardware = hardware or load_hardware()
    profile = hardware_profile(hardware)
    joints = {joint["name"]: joint for joint in profile["joints"]}
    names = list(ARM_JOINTS) + ["gripper"]
    return {key: [float(joints[name]["MIT"][key]) for name in names]
            for key in ("kp", "kd")}


def uart_gravity(hardware: dict | None = None) -> dict[str, list[float]]:
    """返回 UART 重力前馈的零点、方向和缩放；只包含六个机械臂关节。"""
    gravity = (hardware or load_hardware())["uart"]["gravity"]
    return {key: [float(value) for value in gravity[key]]
            for key in ("joint_offset", "joint_direction", "tau_scale")}


def joint_limits(root: Path | None = None) -> dict[str, dict[str, float]]:
    """从 URDF 读取每个关节的角度、速度和力矩限制。"""
    model = ET.parse((root or description_path()) / "urdf/robot_arm.urdf")
    limits = {}
    for joint in model.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limits[joint.get("name")] = {k: float(limit.get(k)) for k in ("lower", "upper", "effort", "velocity")}
    return limits


def motor_specs(hardware: dict, mounts: dict) -> dict:
    """由同一份电机目录生成 ``MotorModel``/扫描器使用的参数结构。

    仿真负载和 TCP 偏移也在这里合并，确保仿真与硬件参数使用同一来源。
    """
    profile = hardware_profile(hardware)
    catalog = hardware["motor_catalog"]
    specs = {"joints": {}, "motors": {}}
    for joint in profile["joints"]:
        if joint["name"] not in ARM_JOINTS:
            continue
        motor = catalog[joint["model"]]
        name = motor["name"]
        specs["joints"][joint["name"]] = {"motor": name, "rated": motor["rated"], "peak": motor["peak"]}
        specs["motors"][name] = {k: v for k, v in motor.items() if k not in ("name", "rated", "peak")}
    specs.update(hardware["simulation"])
    specs["payload"] = {**specs["payload"], "body": mounts["tcp"]["parent_frame"], "tcp_offset_m": mounts["tcp"]["xyz_m"]}
    return specs


def uart_parameters(hardware: dict | None = None,
                    limits: dict | None = None, mounts: dict | None = None) -> dict:
    """组装固定七轴 UART 的关节 ID、限位和安全上限。

    位置限位来自 URDF，电机峰值来自电机目录，UART 的速度/力矩/kp/kd
    上限来自硬件配置；因此调用方无需手工同步多份列表。
    """
    hardware = hardware or load_hardware()
    limits = limits or joint_limits()
    mounts = mounts or load_mounts()
    profile = hardware_profile(hardware)
    joints = {j["name"]: j for j in profile["joints"]}
    names = list(ARM_JOINTS) + ["gripper"]
    policy = hardware["uart"]
    firmware = policy["firmware"]
    result = dict(policy["transport"])
    result["feedback_zero_tolerance_rad"] = [
        float(value) for value in result.get("feedback_zero_tolerance_rad", [0.] * 6)
    ]
    # The MCU frame is ordered by firmware motor groups, while ROS remains in
    # logical joint order. Keep this permutation in the canonical source YAML.
    result["slot_joint_ids"] = [int(value) for value in result["slot_joint_ids"]]
    result["slot_feedback_ids"] = [int(value) for value in result["slot_feedback_ids"]]
    slot_index = {joint_id: index for index, joint_id in enumerate(result["slot_joint_ids"])}
    firmware_logical = {
        key: [float(firmware[key][slot_index[joint_id]]) for joint_id in range(1, len(names) + 1)]
        for key in ("velocity", "torque")
    }
    # UART 按槽位发送，CAN ID 由 STM32 映射，不随 SDK 总线 ID 改变。
    result.update(joint_names=names, joint_ids=list(range(1, len(names) + 1)))
    for key in ("position_min", "position_max", "velocity_limit", "torque_limit", "kp_limit", "kd_limit"):
        result[key] = []
    for index, name in enumerate(names):
        joint = joints[name]
        motor = hardware["motor_catalog"][joint["model"]]
        safety = policy["gripper_limits"] if name == "gripper" else policy["arm_limits"]
        # 夹爪的角度来自硬件配置，其余关节的角度来自 URDF。
        if name == "gripper":
            lo, hi = sorted(profile["gripper"]["position_limits"].values())
        else:
            lo, hi = limits[name]["lower"], limits[name]["upper"]
        result["position_min"].append(float(lo))
        result["position_max"].append(float(hi))
        result["velocity_limit"].append(min(float(safety["velocity"]), firmware_logical["velocity"][index]))
        result["torque_limit"].append(min(float(safety.get("torque", motor["peak"])), float(motor["peak"]),
                                          firmware_logical["torque"][index]))
        result["kp_limit"].append(min(float(safety["kp"]), float(firmware["kp"])))
        result["kd_limit"].append(min(float(safety["kd"]), float(firmware["kd"])))
    result.update(gripper_open_rad=float(profile["gripper"]["position_limits"]["open"]),
                  gripper_close_rad=float(profile["gripper"]["position_limits"]["close"]),
                  gripper_finger_travel_m=limits["gripper_joint2"]["upper"])
    return result


def validate(hardware: dict, limits: dict, mounts: dict) -> None:
    """在生成派生文件前校验三份配置的一致性和物理范围。"""
    profile = hardware_profile(hardware)
    names = [j["name"] for j in profile["joints"]]
    ids = [j["motor_id"] for j in profile["joints"]]
    feedback_ids = [j["feedback_id"] for j in profile["joints"]]
    if (names != list(ARM_JOINTS) + ["gripper"] or
            not all(isinstance(i, int) and 0 < i <= 255 for i in ids) or
            not all(isinstance(i, int) and 0 < i <= 0x7ff for i in feedback_ids)):
        raise ValueError("hardware map must contain seven valid joint and feedback IDs")
    # Local motor IDs may repeat across the separate Damiao and Motorevo CAN
    # buses. A vendor/ID/feedback triple must still identify one joint.
    triples = [(j["vendor"], j["motor_id"], j["feedback_id"]) for j in profile["joints"]]
    if len(set(triples)) != len(triples):
        raise ValueError("hardware map contains duplicate vendor/motor/feedback ID triples")
    if profile['groups']['arm']['joints'] != list(ARM_JOINTS) or profile['groups']['gripper']['joints'] != ['gripper']:
        raise ValueError("SDK joint groups must match the canonical motor map")
    if any(key in profile.get('control', {}) for key in ('mit_kp', 'mit_kd')):
        raise ValueError("set MIT gains on each joint; control.mit_kp/mit_kd would duplicate them")
    for name, limit in limits.items():
        if not all(math.isfinite(v) for v in limit.values()) or limit["lower"] >= limit["upper"] or min(limit["effort"], limit["velocity"]) <= 0:
            raise ValueError(f"invalid URDF limits for {name}")
    for joint in profile["joints"]:
        spec = hardware["motor_catalog"][joint["model"]]
        if any(not math.isfinite(float(v)) or float(v) < 0 for k, v in spec.items() if k != 'name'):
            raise ValueError(f"invalid motor catalog for {joint['name']}")
        if not 0 < float(spec["rated"]) <= float(spec["peak"]):
            raise ValueError(f"invalid torque capabilities for {joint['name']}")
        for key in ("kp", "kd"):
            value = float(joint["MIT"][key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key} for {joint['name']}")
    gripper = profile['gripper']
    angles = [float(v) for v in gripper['position_limits'].values()]
    if len(angles) != 2 or not all(math.isfinite(v) for v in angles) or angles[0] == angles[1]:
        raise ValueError("invalid gripper motor position limits")
    torque = float(gripper['torque_limit_nm'])
    gripper_motor = hardware['motor_catalog'][profile['joints'][-1]['model']]
    if not math.isfinite(torque) or not 0 <= torque <= float(gripper_motor['peak']):
        raise ValueError("invalid measured gripper torque limit")
    for group in ('arm_limits', 'gripper_limits'):
        safety = hardware['uart'][group]
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in safety.values()):
            raise ValueError(f"invalid UART {group}")
    firmware = hardware['uart']['firmware']
    transport = hardware['uart']['transport']
    zero_tolerances = transport.get('feedback_zero_tolerance_rad', [0.] * 6)
    if (len(zero_tolerances) != 6 or
            any(not math.isfinite(float(v)) or float(v) < 0 for v in zero_tolerances)):
        raise ValueError('UART feedback_zero_tolerance_rad must contain six nonnegative finite values')
    slot_joint_ids = transport.get('slot_joint_ids', [])
    slot_motor_ids = transport.get('feedback_motor_ids', [])
    slot_feedback_ids = transport.get('slot_feedback_ids', [])
    if (len(slot_joint_ids) != 7 or sorted(int(v) for v in slot_joint_ids) != list(range(1, 8)) or
            len(slot_motor_ids) != 7 or any(not 1 <= int(v) <= 255 for v in slot_motor_ids) or
            len(slot_feedback_ids) != 7 or any(not 1 <= int(v) <= 0x7ff for v in slot_feedback_ids)):
        raise ValueError('UART slot mapping must contain seven valid logical, motor and feedback IDs')
    for slot_index, logical_id in enumerate(int(v) for v in slot_joint_ids):
        joint = profile['joints'][logical_id - 1]
        if (int(joint['motor_id']) != int(slot_motor_ids[slot_index]) or
                int(joint['feedback_id']) != int(slot_feedback_ids[slot_index])):
            raise ValueError(f"UART slot {slot_index + 1} IDs do not match {joint['name']} hardware IDs")
    for key in ('velocity', 'torque'):
        values = firmware[key]
        if len(values) != 7 or any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError(f"UART firmware {key} must contain seven nonnegative values")
    if not 0 < firmware['kp'] <= 65.535 or not 0 < firmware['kd'] <= 5:
        raise ValueError('UART firmware gain limits exceed the STM32 protocol')
    if any(key in firmware for key in ('default_kp', 'default_kd')):
        raise ValueError('set MIT gains on each joint; firmware.default_kp/default_kd would duplicate them')
    gains = uart_gains(hardware)
    if any(value > float(firmware[key]) for key in ('kp', 'kd') for value in gains[key]):
        raise ValueError('joint MIT gains exceed firmware limits')
    gravity = hardware['uart'].get('gravity', {})
    for key in ('joint_offset', 'joint_direction', 'tau_scale'):
        values = gravity.get(key, [])
        if len(values) != 6 or any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f"UART gravity {key} must contain six finite values")
    if any(abs(float(v)) != 1.0 for v in gravity['joint_direction']):
        raise ValueError('UART gravity joint_direction values must be -1 or 1')
    if any(float(v) < 0 or float(v) > 1 for v in gravity['tau_scale']):
        raise ValueError('UART gravity tau_scale values must be between 0 and 1')
    for joint in profile['joints']:
        safety = hardware['uart']['gripper_limits' if joint['name'] == 'gripper' else 'arm_limits']
        if any(float(joint['MIT'][key]) > float(safety[key]) for key in ('kp', 'kd')):
            raise ValueError(f"MIT gains exceed UART safety ceilings for {joint['name']}")
        if joint['name'] in ARM_JOINTS and float(safety['velocity']) > limits[joint['name']]['velocity']:
            raise ValueError(f"UART velocity exceeds URDF ceiling for {joint['name']}")
    left, right = limits["gripper_joint1"], limits["gripper_joint2"]
    if not (left["upper"] == 0 and right["lower"] == 0 and left["lower"] == -right["upper"]):
        raise ValueError("gripper finger limits must be mirrored")
    if mounts["tcp"]["parent_frame"] != "gripper_end" or mounts["d435_mount"]["parent_frame"] != "camera_link":
        raise ValueError("TCP must attach to gripper_end and sensor to CAD camera_link")
    for frame in ("tcp", "d435_mount"):
        for key in ("xyz_m", "rpy_rad"):
            if len(mounts[frame][key]) != 3 or not all(math.isfinite(float(v)) for v in mounts[frame][key]):
                raise ValueError(f"invalid {frame}.{key}")
    if any(mounts["tcp"]["rpy_rad"]):
        raise ValueError("shared MuJoCo IK currently requires zero TCP local rotation")
