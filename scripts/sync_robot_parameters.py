#!/usr/bin/env python3
"""Synchronize derived robot parameters; --check reports drift without writing.

Sources: description URDF + config/rebotarm_hardware.yaml + d435_mount.yaml.
Only model parameters in MJCF are owned here. Collision/contact geometries,
meshes, solver settings and task/MoveIt policies are maintained separately.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rebotarm_description"))
from rebotarm_description.parameters import (  # noqa: E402
    ARM_JOINTS, uart_gains, uart_gravity, hardware_profile, joint_limits, load_hardware,
    load_mounts, motor_specs, uart_parameters, validate,
)
from rebotarm_description.pingpong_model import derived_pingpong_files  # noqa: E402

YAML_HEADER = "# Derived by scripts/sync_robot_parameters.py; do not edit these values here.\n# Sources: rebotarm_description/urdf/robot_arm.urdf and config/*.yaml.\n"
MJCF_HEADER = "<!-- Parameters synchronized by scripts/sync_robot_parameters.py.\n     Edit URDF/config for joint/body/camera/TCP parameters.\n     Meshes, collision proxies, contact and solver settings remain editable here. -->"


def vector(values) -> str:
    return " ".join(format(float(v), ".15g") for v in values)


def numbers(value: str) -> np.ndarray:
    return np.array([float(v) for v in value.split()])


def rotation(rpy: str) -> Rotation:
    # URDF RPY is fixed-axis roll/pitch/yaw. MuJoCo's lowercase xyz Euler
    # convention differs; quaternions avoid the previous manual conversion.
    return Rotation.from_euler("xyz", numbers(rpy))


def set_pose(element: ET.Element, xyz: str, rpy: str) -> None:
    element.set("pos", xyz)
    for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        element.attrib.pop(attr, None)
    xyzw = rotation(rpy).as_quat()
    element.set("quat", vector(xyzw[[3, 0, 1, 2]]))


def sync_mjcf(description: Path, limits: dict, mounts: dict) -> str:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(description / "mjcf/robot.xml", parser=parser)
    model = tree.getroot()
    urdf = ET.parse(description / "urdf/robot_arm.urdf").getroot()
    bodies = {b.get("name"): b for b in model.findall(".//body")}
    parents = {child: parent for parent in model.iter() for child in parent}
    for source in urdf.findall("joint"):
        child = source.find("child").get("link")
        parent = source.find("parent").get("link")
        body = bodies[child]
        if parents[body].get("name") != parent:
            raise ValueError(f"MJCF topology for {child} differs from URDF; update the model structure first")
        origin = source.find("origin")
        set_pose(body, origin.get("xyz", "0 0 0"), origin.get("rpy", "0 0 0"))
        if source.get("type") == "fixed":
            continue
        name = source.get("name")
        joint = body.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"MJCF is missing joint {name}")
        expected = {"revolute": "hinge", "prismatic": "slide"}.get(source.get("type"))
        if joint.get("type") != expected:
            raise ValueError(f"joint {name} has incompatible URDF/MJCF types")
        joint.set("range", vector([limits[name]["lower"], limits[name]["upper"]]))
        joint.set("axis", source.find("axis").get("xyz"))
        joint.set("limited", "true")
    for link in urdf.findall("link"):
        source = link.find("inertial")
        if source is None:
            continue
        target = bodies[link.get("name")].find("inertial")
        if target is None:
            raise ValueError(f"MJCF is missing inertia for {link.get('name')}")
        origin = source.find("origin")
        inertia = source.find("inertia")
        ix, iy, iz, ixy, ixz, iyz = [float(inertia.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        tensor = np.array([[ix, ixy, ixz], [ixy, iy, iyz], [ixz, iyz, iz]])
        rot = rotation(origin.get("rpy", "0 0 0")).as_matrix()
        tensor = rot @ tensor @ rot.T
        mass = float(source.find("mass").get("value"))
        if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(tensor)) or np.linalg.eigvalsh(tensor).min() <= 0:
            raise ValueError(f"invalid mass/inertia for {link.get('name')}")
        target.attrib.clear()
        target.set("pos", origin.get("xyz", "0 0 0"))
        target.set("mass", vector([mass]))
        target.set("fullinertia", vector([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
    # The master slide is a simulator-only coupling, not the PA43 output angle.
    master = bodies["gripper_coupler"].find("joint[@name='joint7']")
    master.set("range", vector([0, limits["gripper_joint2"]["upper"]]))
    for name in ARM_JOINTS:
        effort = limits[name]["effort"]
        model.find(f"actuator/motor[@joint='{name}']").set("ctrlrange", vector([-effort, effort]))
    finger_effort = min(limits[name]['effort'] for name in ('gripper_joint1', 'gripper_joint2'))
    model.find("actuator/motor[@joint='joint7']").set("ctrlrange", vector([-finger_effort, finger_effort]))
    camera = mounts["d435_mount"]
    # CAD housing pose/mass belong to the URDF. The massless simulator alias
    # only carries the sensor frame; adding inertia here would count it twice.
    d435 = bodies["d435_link"]
    if parents[d435].get("name") != camera["parent_frame"]:
        raise ValueError("MJCF sensor parent differs from d435_mount.yaml")
    set_pose(d435, vector(camera["xyz_m"]), vector(camera["rpy_rad"]))
    tcp = mounts["tcp"]
    hand = bodies[tcp["parent_frame"]]
    site = hand.find("site[@name='gripper_tcp']")
    if site is None:
        site = ET.SubElement(hand, "site", {"name": "gripper_tcp", "size": "0.001", "rgba": "0 0 0 0"})
    set_pose(site, vector(tcp["xyz_m"]), vector(tcp["rpy_rad"]))
    ET.indent(tree, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + MJCF_HEADER + "\n" + ET.tostring(model, encoding="unicode") + "\n"


def yaml_text(data: dict) -> str:
    return YAML_HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def derived_files(root: Path) -> dict[Path, str]:
    description = root / "src/rebotarm_description"
    hardware = load_hardware(description / "config/rebotarm_hardware.yaml")
    mounts = load_mounts(description)
    limits = joint_limits(description)
    validate(hardware, limits, mounts)
    profile = hardware_profile(hardware)
    gains = uart_gains(hardware)
    gravity = uart_gravity(hardware)
    specs = motor_specs(hardware, mounts)
    outputs = {
        root / "src/rebotarm_bringup/config/rebotarm_hardware.yaml": yaml_text({"default_model": hardware["default_model"], "models": hardware["models"]}),
        description / "config/motor_specs.yaml": yaml_text(specs),
        root / "src/rebotarm_mujoco_rs/config/motor_specs.yaml": yaml_text(specs),
        description / "mjcf/robot.xml": sync_mjcf(description, limits, mounts),
    }
    pingpong = yaml.safe_load((description / "config/pingpong.yaml").read_text())
    for filename, text in derived_pingpong_files(outputs[description / "mjcf/robot.xml"], pingpong, hardware, limits).items():
        outputs[description / "mjcf" / filename] = text
    for filename in ("uart.yaml", "uart_grasp.yaml"):
        # 两个 launch 配置共用相同参数，线格式均为七轴 72/131 字节。
        bridge = uart_parameters(hardware, limits, mounts)
        adapter = {
            "joint_names": list(bridge["joint_names"]), "joint_ids": list(bridge["joint_ids"]),
            "arm_kp": gains["kp"][:6], "arm_kd": gains["kd"][:6],
            "arm_velocity_limit": bridge["velocity_limit"][:6],
            "gravity_joint_offset": gravity["joint_offset"],
            "gravity_joint_direction": gravity["joint_direction"],
            "gravity_tau_scale": gravity["tau_scale"],
            "gravity_torque_limit_nm": bridge["torque_limit"][:6],
            "use_joint_gains": True, "gripper_enabled": True,
            "gripper_open_rad": float(profile["gripper"]["position_limits"]["open"]),
            "gripper_close_rad": float(profile["gripper"]["position_limits"]["close"]),
            "gripper_kp": gains["kp"][6], "gripper_kd": gains["kd"][6],
            "gripper_torque_limit_nm": float(profile["gripper"]["torque_limit_nm"]),
            # 固件每次复位都会设置当前位置为零，标定到 URDF 前关闭模型前馈。
            "gravity_feedforward": False,
        }
        outputs[root / "src/rebotarm_uart/config" / filename] = yaml_text({
            "rebotarm_uart_bridge": {"ros__parameters": bridge},
            "rebotarm_posvel_to_mit": {"ros__parameters": copy.deepcopy(adapter)},
        })
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any derived parameter file is stale; do not write")
    args = parser.parse_args()
    # Compute/validate every output before writing any file.
    outputs = derived_files(ROOT)
    stale = [path for path, text in outputs.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
    for path in stale:
        print(("STALE " if args.check else "SYNC ") + str(path.relative_to(ROOT)))
        if not args.check:
            path.write_text(outputs[path], encoding="utf-8")
    if args.check and stale:
        print("Run: python3 scripts/sync_robot_parameters.py", file=sys.stderr)
        return 1
    print(f"Robot parameters {'consistent' if args.check else 'synchronized'} ({len(outputs)} derived files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
