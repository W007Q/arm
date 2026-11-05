"""Unit-aware gripper adapters and shared completion detection, without ROS."""
from dataclasses import dataclass, asdict
import math
import time

@dataclass(frozen=True)
class GripperState:
    backend: str
    measured_gap_m: float
    gap_speed_mps: float
    effort: float
    effort_unit: str
    fault: str = ''
    def payload(self):
        return asdict(self)

class MujocoGripperAdapter:
    def __init__(self, settings):
        self.settings = settings
    def convert(self, positions, velocities, efforts):
        left, right = positions['gripper_joint1'], positions['gripper_joint2']
        speed = max((abs(velocities.get(k, math.nan)) for k in ('gripper_joint1', 'gripper_joint2')), default=math.nan)
        return GripperState('mujoco', self.settings.base_gap_m + right-left,
                            2*speed, max(abs(efforts.get(k, math.nan)) for k in ('gripper_joint1', 'gripper_joint2')), 'N')

class Pa43GripperAdapter:
    def __init__(self, settings, backend='sdk'):
        self.settings, self.backend = settings, backend
    def convert(self, position, velocity, torque, status=0):
        s = self.settings
        scale = s.total_travel_m / (s.motor_open_rad-s.motor_close_rad)
        # Do not clamp measured gap: an encoder/calibration error must remain visible.
        return GripperState(self.backend, s.base_gap_m + (position-s.motor_close_rad)*scale,
                            abs(velocity*scale), abs(torque), 'Nm',
                            f'motor status {status}' if status else '')


def wait_gripper(snapshot, command_mm, settings, *, allow_contact=False,
                 timeout=10., clock=time.monotonic, sleep=time.sleep):
    s = settings
    target_gap = s.base_gap_m + command_mm/s.command_open_mm*s.total_travel_m
    deadline = clock()+timeout
    accepted_since = None
    accepted_kind = None
    samples = []
    initial_gap = None
    last = None
    while clock() < deadline:
        now = clock()
        state, age = snapshot()
        if state is None or age > s.feedback_timeout_sec:
            return {'ok': False, 'message': 'missing or stale gripper feedback'}
        last = state
        if state.get('fault'):
            return {'ok': False, 'message': state['fault']}
        if state.get('backend') != s.backend:
            return {'ok': False, 'message': 'gripper backend does not match task backend'}
        gap, speed, effort = (float(state.get(k, math.nan)) for k in ('measured_gap_m', 'gap_speed_mps', 'effort'))
        if not math.isfinite(gap) or not s.base_gap_m-s.gap_tolerance_m <= gap <= s.maximum_gap_m+s.gap_tolerance_m:
            return {'ok': False, 'message': 'invalid measured gripper gap'}
        if initial_gap is None:
            initial_gap = gap
        samples.append((now, gap))
        samples = [(t, g) for t, g in samples if now-t <= .3]
        stable = speed < s.stable_gap_speed_mps if math.isfinite(speed) else (len(samples)>1 and now-samples[0][0]>=.2 and max(g for _,g in samples)-min(g for _,g in samples)<.001)
        sim = s.backend == 'mujoco'
        threshold = s.sim_contact_force_n if sim else s.hardware_contact_torque_nm
        unit = 'N' if sim else 'Nm'
        reached = abs(gap-target_gap) <= s.gap_tolerance_m
        contact = (allow_contact and threshold > 0 and math.isfinite(effort)
                   and state.get('effort_unit') == unit and effort > threshold
                   and gap > s.base_gap_m + s.contact_endstop_clearance_m
                   and (initial_gap-gap > s.minimum_contact_travel_m or gap > target_gap+s.gap_tolerance_m))
        # Closing to an empty target/endstop never proves a pinch.
        kind = ('contact' if contact else None) if allow_contact else ('target' if reached else None)
        if stable and kind:
            if kind != accepted_kind:
                accepted_kind, accepted_since = kind, now
            elif accepted_since is not None and now-accepted_since >= s.settle_sec:
                return {'ok': True, 'contact_or_target': kind, 'reached': reached,
                        'stalled': contact and not reached, 'contact_candidate': contact,
                        'commanded_gap_m': target_gap, **state}
        else:
            accepted_kind, accepted_since = None, None
        sleep(.05)
    return {'ok': False, 'message': 'gripper did not settle with required evidence before timeout',
            'commanded_gap_m': target_gap, 'last_state': last}
