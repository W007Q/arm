"""Pick geometry. All task poses refer to the same calibrated gripper TCP."""
from __future__ import annotations
import math
from typing import Any
from geometry_msgs.msg import Pose
class PickGeometry:
    def _build_pick_plan(
        self,
        target: dict[str, Any],
        *,
        requested_approach_z: float,
        requested_grasp_z: float,
        requested_lift_z: float,
    ) -> dict[str, Any]:
        x = self._finite_float(target.get("x"), 0.0)
        y = self._finite_float(target.get("y"), 0.0)
        color = self._target_color(target)
        grasp = self._estimate_grasp_plan(target)
        yaw_rad = float(grasp["yaw_rad"])

        detected_z = self._finite_float(target.get("z"), 0.122)
        # Real RGB-D detections report the visible top surface.  Keep the TCP
        # a small distance above it; simulation detections already publish the
        # calibrated object surface in world coordinates.
        source = str(target.get("source") or "").lower()
        if source in {"camera_rgbd", "yolo_rgbd", "realsense_rgbd"}:
            # The RS finger tips extend ~40 mm below this TCP in top-down grasp.
            # Keep TCP 5 mm above the visible surface to clear the table.
            detected_z += self.settings.surface_tcp_clearance_m
        safe_grasp_z = max(
            self.settings.safe_grasp_z_by_color.get(color, 0.122), detected_z
        )
        grasp_z = self._clamp(max(float(requested_grasp_z), safe_grasp_z), 0.06, 0.25)
        transit_floor = self.settings.transit_z_by_color.get(color, self.settings.transit_z_m)
        approach_z = self._clamp(
            max(float(requested_approach_z), transit_floor),
            0.08,
            0.35,
        )
        vertical_align_z = self._clamp(
            max(grasp_z + self.settings.align_clearance_m, self.settings.min_align_z_m),
            grasp_z,
            approach_z,
        )
        pregrasp_z = self._clamp(
            max(grasp_z + self.settings.pregrasp_clearance_m, grasp_z),
            grasp_z,
            approach_z,
        )
        first_lift_floor = self.settings.first_lift_min_m
        first_lift_z = self._clamp(
            max(grasp_z + self.settings.first_lift_clearance_m, first_lift_floor),
            grasp_z,
            approach_z,
        )
        lift_z = self._clamp(
            max(float(requested_lift_z), first_lift_z, self.settings.lift_z_m),
            first_lift_z,
            0.35,
        )

        radius = math.hypot(x, y)
        lift_scale = max(0., 1.-self.settings.lift_retract_m/max(radius, 1e-6))
        return {
            "x": x,
            "y": y,
            "color": color,
            "yaw_rad": yaw_rad,
            "approach_z": approach_z,
            "grasp_z": grasp_z,
            "first_lift_z": first_lift_z,
            "lift_z": lift_z,
            "close_opening_mm": float(grasp["command_mm"]),
            "preopen_opening_mm": self._clamp(
                self._physical_gap_to_gripper_command(
                    float(grasp["cross_section_m"]) + self.settings.preopen_clearance_m
                ) * 1000.0,
                0.0,
                self.settings.command_open_mm,
            ),
            "close_max_effort": float(grasp["max_effort"]),
            "physical_gap_mm": float(grasp["physical_gap_m"]) * 1000.0,
            "transit_pose": self._top_down_pose(x, y, approach_z, yaw_rad),
            "vertical_align_pose": self._top_down_pose(x, y, vertical_align_z, yaw_rad),
            "pregrasp_pose": self._top_down_pose(x, y, pregrasp_z, yaw_rad),
            "grasp_pose": self._top_down_pose(x, y, grasp_z, yaw_rad),
            "first_lift_pose": self._top_down_pose(x, y, first_lift_z, yaw_rad),
            "lift_pose": self._lift_pose(x*lift_scale, y*lift_scale, lift_z, yaw_rad),
        }


    @staticmethod
    def _target_shifted(
        left: dict[str, Any],
        right: dict[str, Any],
        threshold: float,
    ) -> bool:
        if PickGeometry._target_color(left) != PickGeometry._target_color(right):
            return False
        dx = PickGeometry._finite_float(left.get("x"), math.nan) - PickGeometry._finite_float(
            right.get("x"),
            math.nan,
        )
        dy = PickGeometry._finite_float(left.get("y"), math.nan) - PickGeometry._finite_float(
            right.get("y"),
            math.nan,
        )
        dz = PickGeometry._finite_float(left.get("z"), 0.0) - PickGeometry._finite_float(
            right.get("z"),
            0.0,
        )
        if not all(math.isfinite(value) for value in (dx, dy, dz)):
            return False
        return math.hypot(dx, dy, dz) > float(threshold)


    @staticmethod
    def _detection_is_comparable(
        reference: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        """Reject partial contours caused by the wrist hiding the target."""
        reference_short = PickGeometry._finite_float(
            reference.get("shortest_m"), 0.0
        )
        candidate_short = PickGeometry._finite_float(
            candidate.get("shortest_m"), 0.0
        )
        reference_pixels = PickGeometry._finite_float(
            reference.get("pixels"), 0.0
        )
        candidate_pixels = PickGeometry._finite_float(
            candidate.get("pixels"), 0.0
        )
        size_ok = reference_short <= 0.0 or candidate_short >= 0.8 * reference_short
        pixels_ok = (
            reference_pixels <= 0.0
            or candidate_pixels >= 0.75 * reference_pixels
        )
        return size_ok and pixels_ok


    @staticmethod
    def _target_color(target: dict[str, Any]) -> str:
        return str(target.get("color") or "").strip().lower()


    def _estimate_grasp_plan(self, target: dict[str, Any]) -> dict[str, float]:
        fallback_by_color = {
            "red": 0.045,
            "yellow": 0.040,
            "blue": 0.032,
        }
        color = PickGeometry._target_color(target)
        width = PickGeometry._finite_float(target.get("width_m"), math.nan)
        height = PickGeometry._finite_float(target.get("height_m"), math.nan)
        shortest = PickGeometry._finite_float(target.get("shortest_m"), math.nan)
        yaw_rad = 0.0

        if width > 0 and height > 0:
            candidates = [
                {"cross_section": height, "yaw_rad": 0.0},
                {"cross_section": width, "yaw_rad": -math.pi / 2.0},
            ]
            candidates.sort(
                key=lambda item: (
                    item["cross_section"] > self.settings.maximum_gap_m,
                    item["cross_section"],
                )
            )
            cross_section = float(candidates[0]["cross_section"])
            yaw_rad = float(candidates[0]["yaw_rad"])
        else:
            cross_section = shortest
            if not math.isfinite(cross_section) or cross_section <= 0:
                cross_section = fallback_by_color.get(color, 0.05)
            reported_yaw = PickGeometry._finite_float(
                target.get("grasp_yaw_rad"),
                math.nan,
            )
            if math.isfinite(reported_yaw):
                # Parallel jaws are symmetric under pi; avoid unnecessary wrist turns.
                yaw_rad = (reported_yaw + math.pi/2.) % math.pi - math.pi/2.
            longest = PickGeometry._finite_float(
                target.get("longest_m"), math.nan
            )
            # A nearly square object has no preferred in-plane angle.  Keep
            # yaw zero to avoid an unnecessary wrist rotation near J4 limits.
            if (
                math.isfinite(longest)
                and math.isfinite(shortest)
                and shortest > 0
                and longest / shortest < self.settings.yaw_aspect_min_ratio
            ):
                yaw_rad = 0.0

        is_mujoco = self.settings.backend == "mujoco"
        squeeze = self.settings.squeeze_m
        physical_gap = PickGeometry._clamp(
            float(cross_section) - squeeze,
            self.settings.minimum_object_m,
            self.settings.maximum_gap_m,
        )
        command_m = self._physical_gap_to_gripper_command(physical_gap)
        return {
            "command_mm": command_m * 1000.0,
            "cross_section_m": float(cross_section),
            "physical_gap_m": physical_gap,
            "yaw_rad": yaw_rad,
            "max_effort": self.settings.sim_force_limit_n if is_mujoco else self.settings.hardware_torque_limit_nm,
        }


    def _physical_gap_to_gripper_command(self, physical_gap_m: float) -> float:
        travel = max(self.settings.total_travel_m, 0.001)
        open_m = self.settings.command_open_mm / 1000.0
        return PickGeometry._clamp(
            ((float(physical_gap_m) - self.settings.base_gap_m) / travel) * open_m,
            0.0 / 1000.0,
            open_m,
        )


    def _top_down_pose(self, x: float, y: float, z: float, yaw_rad: float) -> Pose:
        yaw = yaw_rad if math.isfinite(float(yaw_rad)) else 0.0
        sy = math.sin(yaw / 2.0)
        cy = math.cos(yaw / 2.0)
        x0, y0, z0, w0 = self.settings.top_down_quat_xyzw
        # q = q_z(yaw) * q0 (rotate the base orientation about world Z).
        return PickGeometry._pose_from_xyz_quat(
            x,
            y,
            z,
            qx=cy * x0 - sy * y0,
            qy=cy * y0 + sy * x0,
            qz=cy * z0 + sy * w0,
            qw=cy * w0 - sy * z0,
        )


    def _lift_pose(self, x: float, y: float, z: float, yaw_rad: float) -> Pose:
        """Keep the pinched object vertical while lifting."""
        return self._top_down_pose(x, y, z, yaw_rad)


    @staticmethod
    def _pose_from_xyz_quat(
        x: float,
        y: float,
        z: float,
        *,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> Pose:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose


    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return number


    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(float(minimum), min(float(maximum), float(value)))

