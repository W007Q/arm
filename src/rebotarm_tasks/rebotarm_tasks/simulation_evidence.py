"""Additional simulation evidence; shared lift verification remains in PickTask."""
from __future__ import annotations
import math
import time
from typing import Any
from .pick_geometry import PickGeometry

class SimulationEvidence(PickGeometry):
    def __init__(self, port):
        self.port = port

    def snapshot(self, color: str) -> dict[str, Any] | None:
        object_name = {
            "red": "red_cube",
            "blue": "blue_block",
            "yellow": "yellow_cylinder",
        }.get(str(color or "").strip().lower())
        if object_name is None:
            return None
        with self.port._lock:
            if time.monotonic() - self.port._scene_payload_at > 0.5:
                return None
            objects = list((self.port._scene_payload or {}).get("objects") or [])
        for item in objects:
            if isinstance(item, dict) and item.get("name") == object_name:
                return dict(item)
        return None


    def wait_contact(
        self,
        target: dict[str, Any],
        baseline: dict[str, Any] | None,
        *,
        timeout_sec: float,
        require_lift: bool = False,
    ) -> dict[str, Any]:
        """Require physical contact from both simulated fingers before/after lift."""
        source = str(target.get("source") or "")
        if not source.startswith("mujoco"):
            return {
                "ok": True,
                "tool": "pick_color.verify_two_finger_contact",
                "skipped": True,
                "message": "MuJoCo contact telemetry is unavailable on hardware.",
            }
        color = self._target_color(target)
        if baseline is None:
            return {
                "ok": False,
                "tool": "pick_color.verify_two_finger_contact",
                "message": "Fresh MuJoCo object state was unavailable before grasping.",
            }
        baseline_position = list(baseline.get("position") or [])
        if len(baseline_position) != 3:
            return {
                "ok": False,
                "tool": "pick_color.verify_two_finger_contact",
                "message": "MuJoCo object state did not contain a valid baseline position.",
            }
        baseline_z = self._finite_float(baseline_position[2], math.nan)
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        stable_since = None
        last = None
        required = {"gripper_left", "gripper_right"}
        while True:
            item = self.snapshot(color)
            if item is not None:
                last = item
                position = list(item.get("position") or [])
                current_z = (
                    self._finite_float(position[2], math.nan)
                    if len(position) == 3
                    else math.nan
                )
                lift_delta = current_z - baseline_z
                contacts = set(item.get("finger_contacts") or [])
                both = required.issubset(contacts)
                if not require_lift and math.isfinite(lift_delta) and lift_delta > 0.008 and not both:
                    return {
                        "ok": False,
                        "tool": "pick_color.verify_two_finger_contact",
                        "message": "Object moved upward before a two-finger pinch was established.",
                        "finger_contacts": sorted(contacts),
                        "premature_lift_m": round(lift_delta, 4),
                        "object": item,
                    }
                height_ok = not require_lift or (
                    math.isfinite(lift_delta) and lift_delta >= 0.02
                )
                if both and height_ok:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= 0.15:
                        return {
                            "ok": True,
                            "tool": "pick_color.verify_two_finger_contact",
                            "finger_contacts": sorted(contacts),
                            "lift_m": round(lift_delta, 4),
                        }
                else:
                    stable_since = None
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        contacts = sorted(set((last or {}).get("finger_contacts") or []))
        return {
            "ok": False,
            "tool": "pick_color.verify_two_finger_contact",
            "message": (
                "Object was not held by both physical finger contacts"
                + (" after lifting." if require_lift else " before lifting.")
            ),
            "finger_contacts": contacts,
            "object": last,
        }


