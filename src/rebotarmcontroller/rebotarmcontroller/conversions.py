#位姿格式转换：在"ROS 的四元数 Pose"和"控制器/IK 用的 xyz+rpy"之间来回转。只做数学转换，没有控制逻辑。
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation


def pose_to_xyz_rpy(pose: Pose) -> tuple[float, float, float, float, float, float]:
    quat = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    roll, pitch, yaw = Rotation.from_quat(quat).as_euler("xyz")
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(roll),
        float(pitch),
        float(yaw),
    )


def fk_to_pose(position: np.ndarray, rotation: np.ndarray) -> Pose:
    quat = Rotation.from_matrix(rotation).as_quat()

    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose
