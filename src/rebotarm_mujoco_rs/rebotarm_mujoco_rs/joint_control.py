"""Joint torque composition shared by the ROS plant and offline experiments."""
import numpy as np

ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))


def pd_bias_torque(q, qd, target, kp, kd, bias, bias_scale=1.0):
    return bias_scale * np.asarray(bias) + kp * (target - q) - kd * qd


def limit_arm_torque(torque, speed, fixed_limits, motor_model=None):
    limits = np.asarray(fixed_limits, dtype=float).copy()
    if motor_model is not None:
        for i, name in enumerate(ARM_JOINTS):
            limits[i] = min(limits[i], motor_model.torque_limit(name, float(speed[i])))
    return np.clip(torque, -limits, limits)
