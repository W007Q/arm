"""Shared colour pick task. Device commands are supplied by the client port."""
from __future__ import annotations
import math
import time
from typing import Any
from geometry_msgs.msg import Pose
from .pick_geometry import PickGeometry
from .simulation_evidence import SimulationEvidence

class PickTask(PickGeometry):
    def __init__(self, port, settings):
        self.port = port
        self.settings = settings
        self.evidence = SimulationEvidence(port)

    def pick_color(
        self,
        color: str = "auto",
        approach_z: float = 0.18,
        grasp_z: float = 0.140,
        lift_z: float = 0.18,
        duration: float = 1.4,
    ) -> dict[str, Any]:
        blocked = self.port._motion_blocked("pick_color")
        if blocked is not None:
            return blocked

        if self.settings.backend != 'mujoco' and not self.settings.hardware_calibrated:
            return {"ok": False, "tool": "pick_color", "message": "hardware calibration is incomplete; set measured TCP, D435 and PA43 thresholds in the hardware profile"}
        detection_result = self._acquire_pick_target(color)
        if not detection_result.get("ok"):
            return detection_result
        target = detection_result.get("target")
        if not isinstance(target, dict):
            return {
                "ok": False,
                "tool": "pick_color",
                "message": f"No target found for color={color!r}.",
            }
        source_is_sim = str(target.get('source') or '').startswith('mujoco')
        if source_is_sim != (self.settings.backend == 'mujoco'):
            return {'ok': False, 'message': 'vision source differs from configured task backend'}
        if not all(
            math.isfinite(self._finite_float(target.get(axis), math.nan))
            for axis in ("x", "y", "z")
        ):
            return {
                "ok": False,
                "tool": "pick_color",
                "message": "Target detection is missing finite x, y, or z coordinates.",
                "target": target,
            }

        plan = self._build_pick_plan(
            target,
            requested_approach_z=float(approach_z),
            requested_grasp_z=float(grasp_z),
            requested_lift_z=float(lift_z),
        )
        duration = max(float(duration), 0.3)
        steps = []
        simulation_baseline = self.evidence.snapshot(
            str(target.get("color") or color)
        )

        for label, action in [
            (
                "open_gripper",
                lambda: self.port.set_gripper_opening_mm(
                    self.settings.command_open_mm, guard_motion=False
                ),
            ),
            (
                "wait_open_gripper",
                lambda: self.port.wait_gripper(self.settings.command_open_mm),
            ),
            (
                "high_align",
                lambda: self.port.move_pick_pose(
                    plan["transit_pose"],
                    max(duration, 1.4),
                    tool="pick_color.high_align",
                ),
            ),
            (
                "vertical_align",
                lambda: self.port.move_pick_pose(
                    plan["vertical_align_pose"],
                    max(duration * 0.75, 0.9),
                    tool="pick_color.vertical_align",
                ),
            ),
            (
                "pregrasp",
                lambda: self.port.move_pick_pose(
                    plan["pregrasp_pose"],
                    max(duration * 0.6, 0.8),
                    tool="pick_color.pregrasp",
                ),
            ),
            (
                "move_down",
                lambda: self.port.move_pick_pose(
                    plan["grasp_pose"],
                    max(duration * 0.65, 0.9),
                    tool="pick_color.move_down",
                ),
            ),
            (
                "close_gripper",
                lambda: self.port.set_gripper_opening_mm(
                    plan["close_opening_mm"],
                    max_effort=plan["close_max_effort"],
                    guard_motion=False,
                ),
            ),
            (
                "settle_after_close",
                lambda: self.port.wait_gripper(plan["close_opening_mm"], allow_contact=True),
            ),
            (
                "verify_two_finger_contact",
                lambda: self.evidence.wait_contact(
                    target,
                    simulation_baseline,
                    timeout_sec=1.5,
                ),
            ),
            (
                "first_lift",
                lambda: self.port.move_pick_pose(
                    plan["first_lift_pose"],
                    max(duration * 0.75, 1.1),
                    tool="pick_color.first_lift",
                ),
            ),
            (
                "lift",
                lambda: self.port.move_pick_pose(
                    plan["lift_pose"],
                    max(duration, 1.2),
                    tool="pick_color.lift",
                ),
            ),
        ]:
            if label == "first_lift":
                attachment = self.port.mark_grasp(str(target.get("color") or color))
                steps.append({"step": "attach_payload_for_planning", "result": attachment})
                if not attachment.get("ok"):
                    return {"ok": False, "failed_step": "attach_payload_for_planning", "steps": steps}
            if label == "move_down":
                checked = self._wait_for_detection(
                    str(target.get("color") or color), timeout_sec=0.6,
                    min_received_at=time.monotonic(),
                )
                checked_target = checked.get("target") if checked.get("ok") else None
                if (
                    isinstance(checked_target, dict)
                    and self._detection_is_comparable(target, checked_target)
                    and self._target_shifted(target, checked_target, 0.008)
                ):
                    return {
                        "ok": False, "tool": "pick_color", "target": target,
                        "current_target": checked_target,
                        "failed_step": "recheck_target_before_descent",
                        "message": "Target moved more than 8 mm during approach; descent to the old position is unsafe.",
                        "steps": steps,
                    }
            if label == "vertical_align":
                refined = self._wait_for_detection(
                    str(target.get("color") or color),
                    timeout_sec=0.35,
                )
                refined_target = refined.get("target") if refined.get("ok") else None
                if (isinstance(refined_target, dict)
                    and self._detection_is_comparable(target, refined_target)
                    and self._target_shifted(
                    target, refined_target, 0.008
                )):
                    target = refined_target
                    plan = self._build_pick_plan(
                        target,
                        requested_approach_z=float(approach_z),
                        requested_grasp_z=float(grasp_z),
                        requested_lift_z=float(lift_z),
                    )
            self.port.select_target(str(target.get("color") or color))
            time.sleep(.05)
            result = action()
            steps.append({"step": label, "result": result})
            if not result.get("ok", result.get("ros_success", False)):
                return {
                    "ok": False,
                    "tool": "pick_color",
                    "target": target,
                    "failed_step": label,
                    "steps": steps,
                }

        two_finger_hold = self.evidence.wait_contact(
            target,
            simulation_baseline,
            timeout_sec=0.8,
            require_lift=True,
        )
        steps.append({"step": "verify_two_finger_hold", "result": two_finger_hold})
        if not two_finger_hold.get("ok"):
            return {
                "ok": False,
                "tool": "pick_color",
                "target": target,
                "failed_step": "verify_two_finger_hold",
                "steps": steps,
            }

        # A trajectory result can arrive before the next camera frame. Require
        # a detection received after the lift, otherwise old z data looks like
        # a failed grasp even when the object moved.
        verification = self._wait_for_detection(
            str(target.get("color") or color),
            timeout_sec=1.5,
            min_received_at=time.monotonic(),
        )
        if not verification.get("ok"):
            return {
                "ok": False,
                "tool": "pick_color",
                "target": target,
                "failed_step": "verify_physical_lift",
                "message": verification.get("message", "No fresh post-lift camera frame."),
                "steps": steps,
            }
        verified_target = verification.get("target") if verification.get("ok") else None
        start_z = self._finite_float(target.get("z"), 0.0)
        final_z = (
            self._finite_float(verified_target.get("z"), start_z)
            if isinstance(verified_target, dict)
            else start_z
        )
        lift_delta = final_z - start_z
        if lift_delta < self.settings.lift_verify_m:
            return {
                "ok": False,
                "tool": "pick_color",
                "target": target,
                "failed_step": "verify_physical_lift",
                "message": (
                    "Motion completed, but camera detections did not confirm the object "
                    f"being lifted (delta_z={lift_delta:.4f} m)."
                ),
                "final_target": verified_target,
                "steps": steps,
            }

        # A narrow or high pinch can lift the object for one frame and then
        # let it slide out.  Require another fresh D435 observation after a
        # short hold so a transient lift is not reported as a successful
        # grasp.
        time.sleep(self.settings.hold_sec)
        hold_observation_started = time.monotonic()
        hold_verification = self._wait_for_detection(
            str(target.get("color") or color),
            timeout_sec=1.5,
            min_received_at=hold_observation_started,
        )
        held_target = (
            hold_verification.get("target")
            if hold_verification.get("ok")
            else None
        )
        held_z = (
            self._finite_float(held_target.get("z"), start_z)
            if isinstance(held_target, dict)
            else start_z
        )
        held_lift_delta = held_z-start_z
        if held_lift_delta < self.settings.lift_verify_m:
            return {
                "ok": False,
                "tool": "pick_color",
                "target": target,
                "failed_step": "verify_sustained_grasp",
                "message": (
                    f"The object was initially lifted but was not held for {self.settings.hold_sec:g} s "
                    f"(delta_z={held_lift_delta:.4f} m)."
                ),
                "final_target": held_target,
                "steps": steps,
            }

        return {
            "ok": True,
            "tool": "pick_color",
            "target": target,
            "plan": {
                "yaw_deg": round(math.degrees(float(plan["yaw_rad"])), 1),
                "close_opening_mm": round(float(plan["close_opening_mm"]), 1),
                "close_max_effort": round(float(plan["close_max_effort"]), 1),
                "approach_z": round(float(plan["approach_z"]), 4),
                "grasp_z": round(float(plan["grasp_z"]), 4),
                "first_lift_z": round(float(plan["first_lift_z"]), 4),
                "lift_z": round(float(plan["lift_z"]), 4),
            },
            "steps": steps,
            "verified_lift_m": round(lift_delta, 4),
            "sustained_lift_m": round(held_lift_delta, 4),
            "hold_time_sec": self.settings.hold_sec,
        }


    def _acquire_pick_target(self, color: str) -> dict[str, Any]:
        """Acquire a target with the wrist-mounted D435.

        A fixed overhead camera could see the whole table from any arm pose.
        The D435 cannot, so first accept an already visible fresh target, then
        move through high, collision-checked observation poses when needed.
        """
        existing = self._wait_for_detection(color, timeout_sec=0.25)
        if existing.get("ok") and isinstance(existing.get("target"), dict):
            return existing

        requested = str(color or "auto").strip().lower()
        observation_y = self.settings.observation_y_by_color.get(requested, self.settings.observation_y_by_color['auto'])

        opened = self.port.set_gripper_opening_mm(self.settings.command_open_mm, guard_motion=False)
        if not opened.get("ok", opened.get("ros_success", False)):
            return opened
        settled = self.port.wait_gripper(self.settings.command_open_mm)
        if not settled.get("ok"):
            return settled

        last_result: dict[str, Any] = existing
        for y in observation_y:
            moved = self.port.move_pick_pose(
                self._top_down_pose(self.settings.observation_x_m, float(y), self.settings.observation_z_m, 0.0),
                1.6,
                tool="pick_color.observe",
            )
            if not moved.get("ok", moved.get("ros_success", False)):
                last_result = moved
                continue
            last_result = self._wait_for_detection(
                requested,
                timeout_sec=0.9,
                min_received_at=time.monotonic(),
            )
            if last_result.get("ok") and isinstance(last_result.get("target"), dict):
                return last_result
        return last_result


    def _wait_for_detection(
        self,
        preferred_color: str,
        *,
        timeout_sec: float,
        min_received_at: float = 0.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        last = None
        while True:
            result = self.port.detect_blocks(preferred_color)
            received_at = self.port.vision_received_at()
            if (
                result.get("ok")
                and isinstance(result.get("target"), dict)
                and received_at > min_received_at
            ):
                age = result.get("age_sec")
                if age is None or float(age) <= 0.8:
                    return result
            last = result
            if time.monotonic() >= deadline:
                break
            time.sleep(0.08)
        return {
            "ok": False,
            "tool": "detect_blocks",
            "message": "No fresh target detection was received before timeout.",
            "last_detection": last,
        }


