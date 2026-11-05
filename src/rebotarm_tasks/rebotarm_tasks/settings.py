"""Task calibration and policy, shared by simulation and hardware clients."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import math
import yaml
from rebotarm_description.parameters import hardware_profile, joint_limits, load_hardware


def _motor_angle(endpoint):
    return float(hardware_profile(load_hardware())['gripper']['position_limits'][endpoint])


def _finger_travel():
    limits = joint_limits()
    return limits['gripper_joint2']['upper'] - limits['gripper_joint1']['lower']

@dataclass
class PickSettings:
    backend: str = 'mujoco'
    hardware_calibrated: bool = False
    command_open_mm: float = 90.0  # legacy command scale, not physical jaw gap
    motor_open_rad: float = field(default_factory=lambda: _motor_angle('open'))
    motor_close_rad: float = field(default_factory=lambda: _motor_angle('close'))
    base_gap_m: float = 0.0193
    total_travel_m: float = field(default_factory=_finger_travel)
    minimum_object_m: float = 0.010
    squeeze_m: float = 0.003
    preopen_clearance_m: float = 0.006
    yaw_aspect_min_ratio: float = 1.3
    sim_force_limit_n: float = 2.0
    sim_contact_force_n: float = 0.15
    hardware_torque_limit_nm: float = 0.0  # set only after PA43 calibration
    hardware_contact_torque_nm: float = 0.0
    gap_tolerance_m: float = 0.003
    stable_gap_speed_mps: float = 0.010
    minimum_contact_travel_m: float = 0.004
    contact_endstop_clearance_m: float = 0.003
    settle_sec: float = 0.25
    feedback_timeout_sec: float = 0.5
    lift_verify_m: float = 0.035
    hold_sec: float = 2.0
    top_down_quat_xyzw: list = field(default_factory=lambda: [-0.00113043, 0.27370228, 0.00397238, 0.96180560])
    transit_z_m: float = 0.160
    transit_z_by_color: dict = field(default_factory=lambda: {'red': 0.230, 'blue': 0.210, 'yellow': 0.210})
    safe_grasp_z_by_color: dict = field(default_factory=lambda: {'red': 0.160, 'blue': 0.154, 'yellow': 0.168})
    align_clearance_m: float = 0.050
    min_align_z_m: float = 0.145
    pregrasp_clearance_m: float = 0.040
    first_lift_min_m: float = 0.225
    first_lift_clearance_m: float = 0.020
    lift_z_m: float = 0.240
    lift_retract_m: float = 0.025
    surface_tcp_clearance_m: float = 0.005
    observation_x_m: float = 0.34
    observation_z_m: float = 0.23
    observation_y_by_color: dict = field(default_factory=lambda: {'red': [0.0, -0.12], 'blue': [0.0], 'yellow': [0.14, 0.0], 'auto': [0.0, 0.14, -0.14]})

    @property
    def maximum_gap_m(self):
        return self.base_gap_m + self.total_travel_m

    def validate(self):
        if self.backend not in ('mujoco', 'sdk', 'uart'):
            raise ValueError('backend must be mujoco, sdk or uart')
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{f.name} must be finite')
        if self.motor_open_rad == self.motor_close_rad or min(self.total_travel_m, self.command_open_mm, self.settle_sec, self.feedback_timeout_sec) <= 0:
            raise ValueError('invalid gripper calibration or timing')
        if len(self.top_down_quat_xyzw) != 4 or not all(math.isfinite(x) for x in self.top_down_quat_xyzw) or abs(sum(x*x for x in self.top_down_quat_xyzw)-1) > 0.01:
            raise ValueError('top_down_quat_xyzw must be a unit quaternion')
        if min(self.base_gap_m, self.sim_force_limit_n, self.sim_contact_force_n, self.hardware_torque_limit_nm, self.hardware_contact_torque_nm) < 0:
            raise ValueError('gap and effort limits cannot be negative')
        if self.hardware_calibrated and self.backend != 'mujoco' and not 0 < self.hardware_contact_torque_nm < self.hardware_torque_limit_nm:
            raise ValueError('calibrated hardware needs contact threshold below torque limit')
        if self.yaw_aspect_min_ratio <= 1.:
            raise ValueError('yaw_aspect_min_ratio must exceed one')
        if self.lift_retract_m < 0:
            raise ValueError('lift_retract_m cannot be negative')
        for name in ('transit_z_by_color', 'safe_grasp_z_by_color'):
            values = getattr(self, name)
            if not isinstance(values, dict) or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in values.values()):
                raise ValueError(f'{name} must map colours to finite heights')
        return self


def load_settings(path=''):
    if not path:
        return PickSettings().validate()
    data = yaml.safe_load(Path(path).read_text()) or {}
    data = data.get('pick', data)
    known = {f.name for f in fields(PickSettings)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f'unknown pick settings: {sorted(unknown)}')
    return PickSettings(**data).validate()
