import threading
from types import SimpleNamespace
import pytest

from rebotarm_tasks import pick_task as agent
from rebotarm_tasks.pick_task import PickTask
from rebotarm_tasks.pick_geometry import PickGeometry
from rebotarm_tasks.simulation_evidence import SimulationEvidence
from rebotarm_tasks.settings import PickSettings


def test_wait_for_detection_requires_post_motion_frame(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    bridge = SimpleNamespace(_lock=threading.RLock(), _vision_payload_at=9.9)
    bridge.vision_received_at = lambda: bridge._vision_payload_at
    bridge.detect_blocks = lambda color: {
        "ok": True, "target": {"color": color, "z": 0.12}, "age_sec": 0.1
    }

    def advance(seconds):
        clock.now += seconds
        if clock.now >= 10.24:
            bridge._vision_payload_at = clock.now

    monkeypatch.setattr(
        agent, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=advance)
    )
    result = PickTask(bridge, PickSettings())._wait_for_detection( "red", timeout_sec=0.5, min_received_at=10.0
    )
    assert result["ok"]
    assert clock.now >= 10.24


def test_wait_for_detection_rejects_stale_frame(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    bridge = SimpleNamespace(_lock=threading.RLock(), _vision_payload_at=9.0)
    bridge.vision_received_at = lambda: bridge._vision_payload_at
    bridge.detect_blocks = lambda color: {
        "ok": True, "target": {"color": color, "z": 0.12}, "age_sec": 1.0
    }
    monkeypatch.setattr(
        agent,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    result = PickTask(bridge, PickSettings())._wait_for_detection( "red", timeout_sec=0.2
    )
    assert not result["ok"]
    assert "fresh" in result["message"]


def test_partial_occluded_detection_is_not_comparable():
    reference = {"shortest_m": 0.044, "pixels": 240}
    occluded = {"shortest_m": 0.043, "pixels": 87}
    assert not PickGeometry._detection_is_comparable(reference, occluded)


def test_similar_full_detection_is_comparable():
    reference = {"shortest_m": 0.044, "pixels": 240}
    candidate = {"shortest_m": 0.043, "pixels": 220}
    assert PickGeometry._detection_is_comparable(reference, candidate)


def test_red_grasp_uses_calibrated_two_finger_travel():
    plan = PickTask(SimpleNamespace(), PickSettings())._estimate_grasp_plan({
        "color": "red", "shortest_m": 0.045, "longest_m": 0.045,
        "grasp_yaw_rad": 0.0, "source": "mujoco_rgbd",
    })
    assert 20.0 < plan["command_mm"] < 24.0
    assert plan["yaw_rad"] == 0.0


def test_yolo_rgbd_target_keeps_tcp_above_visible_surface():
    task = PickTask(SimpleNamespace(), PickSettings())
    plan = task._build_pick_plan(
        {"color": "red", "source": "yolo_rgbd", "x": .3, "y": .0, "z": .12,
         "shortest_m": .04, "longest_m": .05},
        requested_approach_z=.18, requested_grasp_z=.12, requested_lift_z=.20,
    )
    assert plan["grasp_z"] >= .125


def test_hardware_detection_skips_simulation_contact_gate():
    result = SimulationEvidence(SimpleNamespace()).wait_contact(
        {"color": "red", "source": "realsense_rgbd"},
        None,
        timeout_sec=0.0,
    )
    assert result["ok"] and result["skipped"]


def test_simulation_rejects_single_finger_premature_lift():
    bridge = SimulationEvidence(SimpleNamespace())
    bridge.snapshot = lambda _color: {
        "name": "red_cube",
        "position": [0.34, -0.12, 0.09],
        "finger_contacts": ["gripper_right"],
    }
    result = bridge.wait_contact(
        {"color": "red", "source": "mujoco_rgbd"},
        {"name": "red_cube", "position": [0.34, -0.12, 0.078]},
        timeout_sec=0.0,
    )
    assert not result["ok"]
    assert result["finger_contacts"] == ["gripper_right"]
    assert result["premature_lift_m"] == pytest.approx(0.012)


def test_top_down_orientation_points_fingers_toward_table():
    from scipy.spatial.transform import Rotation
    q = PickSettings().top_down_quat_xyzw
    rotation = Rotation.from_quat(q).as_matrix()
    # This CAD axis points from the fingertips back toward the wrist.
    finger_to_wrist = [0.0, 0.0, 0.0]
    finger_to_wrist[0] = -0.52650574
    finger_to_wrist[2] = 0.85017157
    world = rotation @ finger_to_wrist
    assert world[2] > 0.999


def test_round_target_orientation_does_not_flip_on_slight_partial_contour():
    task=PickTask(SimpleNamespace(), PickSettings())
    plan=task._estimate_grasp_plan({'color':'yellow','shortest_m':.03337,
                                    'longest_m':.03868,'grasp_yaw_rad':2.099})
    assert plan['yaw_rad']==0.


def test_parallel_jaw_orientation_uses_pi_symmetry_for_elongated_object():
    import math
    task=PickTask(SimpleNamespace(), PickSettings())
    plan=task._estimate_grasp_plan({'color':'blue','shortest_m':.032,
                                    'longest_m':.052,'grasp_yaw_rad':2.099})
    assert plan['yaw_rad']==pytest.approx(2.099-math.pi)
