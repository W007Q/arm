"""Shared ROS task port for CLI and MCP. It contains no language model code."""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rebotarm_msgs.action import MoveToPose
from rebotarm_msgs.msg import ArmStatus, JointMotorState
from rebotarm_msgs.srv import MoveToPoseIK, SetGripper
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from .pick_task import PickTask
from .settings import load_settings
from .gripper import wait_gripper

_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

class RosTaskClient(Node):
    def __init__(
        self,
        *,
        namespace: str,
        motion_allowed: bool,
        service_timeout_sec: float,
        action_timeout_padding_sec: float,
        config_path: str = "",
    ) -> None:
        super().__init__("rebotarm_task_client")
        self.namespace = namespace.strip("/") or "rebotarm"
        self.motion_allowed = bool(motion_allowed)
        self.service_timeout_sec = max(float(service_timeout_sec), 0.1)
        self.action_timeout_padding_sec = max(float(action_timeout_padding_sec), 1.0)

        self.settings = load_settings(config_path)
        self.task = PickTask(self, self.settings)
        self._lock = threading.RLock()
        self._normalized_gripper = None
        self._normalized_gripper_at = 0.0
        self._pick_lock = threading.Lock()
        self._joint_state: dict[str, Any] | None = None
        self._joint_state_at = 0.0
        self._arm_status: dict[str, Any] | None = None
        self._arm_status_at = 0.0
        self._gripper_state: dict[str, Any] | None = None
        self._gripper_state_at = 0.0
        self._vision_payload: dict[str, Any] | None = None
        self._vision_payload_at = 0.0
        self._vision_parse_error: str | None = None
        self._scene_payload: dict[str, Any] | None = None
        self._scene_payload_at = 0.0

        self._trigger_clients = {
            "enable": self.create_client(Trigger, self._srv("enable")),
            "disable": self.create_client(Trigger, self._srv("disable")),
            "safe_home": self.create_client(Trigger, self._srv("safe_home")),
            "gravity_start": self.create_client(
                Trigger, self._srv("gravity_compensation/start")
            ),
            "gravity_stop": self.create_client(
                Trigger, self._srv("gravity_compensation/stop")
            ),
            "gravity_status": self.create_client(
                Trigger, self._srv("gravity_compensation/status")
            ),
            "record_start": self.create_client(Trigger, self._srv("mujoco/record/start")),
            "record_stop": self.create_client(Trigger, self._srv("mujoco/record/stop")),
            "record_replay": self.create_client(
                Trigger, self._srv("mujoco/record/replay")
            ),
            "record_clear": self.create_client(Trigger, self._srv("mujoco/record/clear")),
        }
        self._gripper_client = self.create_client(SetGripper, self._srv("gripper/set"))
        self._ik_client = self.create_client(MoveToPoseIK, self._srv("move_to_pose_ik"))
        self._move_to_pose_action = ActionClient(
            self, MoveToPose, f"/{self.namespace}/move_to_pose"
        )
        self._follow_joint_trajectory_action = ActionClient(
            self, FollowJointTrajectory, f"/{self.namespace}/follow_joint_trajectory"
        )
        self._attach_client = self.create_client(Trigger, self._srv('planning/attach'))
        self._release_client = self.create_client(Trigger, self._srv('planning/release'))
        self._grasp_target_pub = self.create_publisher(String, f"/{self.namespace}/planning/grasp_target", 10)

        self.create_subscription(String, f"/{self.namespace}/gripper/status", self._normalized_gripper_callback, 10)
        latched_status_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(
            JointState,
            f"/{self.namespace}/joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ArmStatus,
            f"/{self.namespace}/arm_status",
            self._arm_status_callback,
            latched_status_qos,
        )
        self.create_subscription(
            JointMotorState,
            f"/{self.namespace}/gripper/state",
            self._gripper_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            f"/{self.namespace}/vision/color_blocks/detections",
            self._vision_callback,
            10,
        )
        self.create_subscription(
            String,
            f"/{self.namespace}/mujoco/object_states",
            self._scene_callback,
            10,
        )


    def _srv(self, name: str) -> str:
        return f"/{self.namespace}/{name}"


    def _joint_state_callback(self, msg: JointState) -> None:
        positions = {}
        velocities = {}
        efforts = {}
        for index, name in enumerate(msg.name):
            if index < len(msg.position):
                positions[name] = float(msg.position[index])
            if index < len(msg.velocity):
                velocities[name] = float(msg.velocity[index])
            if index < len(msg.effort):
                efforts[name] = float(msg.effort[index])
        with self._lock:
            self._joint_state = {
                "names": list(msg.name),
                "positions": positions,
                "velocities": velocities,
                "efforts": efforts,
            }
            self._joint_state_at = time.monotonic()


    def _arm_status_callback(self, msg: ArmStatus) -> None:
        with self._lock:
            self._arm_status = {
                "mode": msg.mode,
                "enabled": bool(msg.enabled),
                "control_loop_active": bool(msg.control_loop_active),
                "state_machine": msg.state_machine,
                "joint_names": list(msg.joint_names),
                "per_joint_status_code": [int(value) for value in msg.per_joint_status_code],
                "error_codes": list(msg.error_codes),
            }
            self._arm_status_at = time.monotonic()


    def _gripper_state_callback(self, msg: JointMotorState) -> None:
        with self._lock:
            self._gripper_state = {
                "joint_name": msg.joint_name,
                "motor_position_rad": float(msg.position),
                "opening_mm": self._motor_rad_to_opening_mm(float(msg.position)),
                "velocity": float(msg.velocity),
                "torque": float(msg.torque),
                "status_code": int(msg.status_code),
            }
            self._gripper_state_at = time.monotonic()


    def _vision_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "{}")
            parse_error = None
        except Exception as exc:
            payload = None
            parse_error = str(exc)
        with self._lock:
            self._vision_payload = payload
            self._vision_parse_error = parse_error
            self._vision_payload_at = time.monotonic()


    def _scene_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
            return
        with self._lock:
            self._scene_payload = payload
            self._scene_payload_at = time.monotonic()


    def get_status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "ok": True,
                "namespace": self.namespace,
                "motion_mode": "allow" if self.motion_allowed else "locked",
                "topics": {
                    "joint_states_age_sec": self._age(now, self._joint_state_at),
                    "arm_status_age_sec": self._age(now, self._arm_status_at),
                    "gripper_state_age_sec": self._age(now, self._gripper_state_at),
                    "vision_age_sec": self._age(now, self._vision_payload_at),
                    "mujoco_scene_age_sec": self._age(now, self._scene_payload_at),
                },
                "arm_status": self._arm_status,
                "joint_state": self._joint_state,
                "gripper_state": self._gripper_state,
                "vision": self._vision_summary_locked(now),
            }


    def diagnose_ros(self) -> dict[str, Any]:
        services = {}
        for name, client in self._trigger_clients.items():
            services[name] = bool(client.wait_for_service(timeout_sec=0.05))
        services["gripper_set"] = bool(
            self._gripper_client.wait_for_service(timeout_sec=0.05)
        )
        services["move_to_pose_ik"] = bool(
            self._ik_client.wait_for_service(timeout_sec=0.05)
        )
        actions = {
            "move_to_pose": bool(
                self._move_to_pose_action.wait_for_server(timeout_sec=0.05)
            ),
            "follow_joint_trajectory": bool(
                self._follow_joint_trajectory_action.wait_for_server(timeout_sec=0.05)
            ),
        }
        status = self.get_status()
        return {
            "ok": True,
            "namespace": self.namespace,
            "motion_mode": status["motion_mode"],
            "services": services,
            "actions": actions,
            "topics": status["topics"],
        }


    def trigger(self, name: str, *, guard_motion: bool = False) -> dict[str, Any]:
        if guard_motion:
            blocked = self._motion_blocked(name)
            if blocked is not None:
                return blocked
        client = self._trigger_clients[name]
        try:
            result = self._call_service(client, Trigger.Request(), self._srv_name(name))
        except Exception as exc:
            return self._tool_error(name, exc)
        return self._trigger_result(name, result)


    def set_gripper_opening_mm(
        self,
        opening_mm: float,
        max_effort: float = 0.0,
        *,
        guard_motion: bool = True,
    ) -> dict[str, Any]:
        if guard_motion:
            blocked = self._motion_blocked("set_gripper_opening_mm")
            if blocked is not None:
                return blocked

        opening_mm = self._clamp(float(opening_mm), 0.0, self.settings.command_open_mm)
        request = SetGripper.Request()
        request.position = self._opening_mm_to_motor_rad(opening_mm)
        request.max_effort = float(max_effort)
        try:
            result = self._call_service(
                self._gripper_client,
                request,
                self._srv("gripper/set"),
            )
        except Exception as exc:
            return self._tool_error("set_gripper_opening_mm", exc)
        if result.success and opening_mm >= self.settings.command_open_mm * .95 and self.settings.backend != 'mujoco':
            released = self._payload_service(self._release_client, 'release')
            if not released.get('ok'):
                return released
        return {
            "ok": bool(result.success),
            "tool": "set_gripper_opening_mm",
            "requested_opening_mm": opening_mm,
            "ros_success": bool(result.success),
            "reached_opening_mm": self._motor_rad_to_opening_mm(
                float(result.reached_position)
            ),
        }


    def ik_check(
        self,
        x: float,
        y: float,
        z: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> dict[str, Any]:
        request = MoveToPoseIK.Request()
        request.target_pose = self._pose_from_xyz_rpy_deg(
            x, y, z, roll_deg, pitch_deg, yaw_deg
        )
        try:
            result = self._call_service(
                self._ik_client,
                request,
                self._srv("move_to_pose_ik"),
            )
        except Exception as exc:
            return self._tool_error("ik_check", exc)
        return {
            "ok": True,
            "tool": "ik_check",
            "ros_success": bool(result.success),
            "message": str(result.message),
            "q_solution": [float(value) for value in result.q_solution],
        }


    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 2.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        *,
        guard_motion: bool = True,
    ) -> dict[str, Any]:
        if guard_motion:
            blocked = self._motion_blocked("move_to_pose")
            if blocked is not None:
                return blocked

        duration = max(float(duration), 0.2)
        if not self._move_to_pose_action.wait_for_server(
            timeout_sec=self.service_timeout_sec
        ):
            return self._unavailable("move_to_pose", f"/{self.namespace}/move_to_pose")

        pose = self._pose_from_xyz_rpy_deg(x, y, z, roll_deg, pitch_deg, yaw_deg)
        return self._send_move_to_pose(pose, duration, tool="move_to_pose")


    def _move_to_pose_with_pose(
        self,
        pose: Pose,
        duration: float,
        *,
        tool: str,
    ) -> dict[str, Any]:
        if not self._move_to_pose_action.wait_for_server(
            timeout_sec=self.service_timeout_sec
        ):
            return self._unavailable(tool, f"/{self.namespace}/move_to_pose")
        return self._send_move_to_pose(pose, max(float(duration), 0.2), tool=tool)


    def _send_move_to_pose(
        self,
        pose: Pose,
        duration: float,
        *,
        tool: str,
    ) -> dict[str, Any]:
        goal = MoveToPose.Goal()
        goal.target_pose = pose
        goal.duration = max(float(duration), 0.2)
        send_future = self._move_to_pose_action.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(
                send_future,
                self.service_timeout_sec,
                f"{tool} send_goal",
            )
        except Exception as exc:
            return self._tool_error(tool, exc)
        if not goal_handle.accepted:
            return {"ok": False, "tool": tool, "message": "goal rejected"}

        result_future = goal_handle.get_result_async()
        try:
            response = self._wait_future(
                result_future,
                duration + self.action_timeout_padding_sec,
                f"{tool} result",
            )
        except Exception as exc:
            goal_handle.cancel_goal_async()
            return self._tool_error(tool, exc)
        result = response.result
        return {
            "ok": bool(result.success),
            "tool": tool,
            "ros_success": bool(result.success),
            "message": str(result.message),
            "final_pose": self._pose_to_dict(result.final_pose),
            "status": int(getattr(response, "status", 0)),
        }


    def _move_to_pose_via_ik_trajectory(
        self,
        pose: Pose,
        duration: float,
        *,
        tool: str,
    ) -> dict[str, Any]:
        ik = self._solve_pose_ik(pose, tool=f"{tool}.ik")
        if not ik.get("ok"):
            return ik

        q_solution = ik.get("q_solution")
        if not isinstance(q_solution, list) or len(q_solution) < len(_ARM_JOINTS):
            return {
                "ok": False,
                "tool": tool,
                "message": "IK did not return a complete joint solution.",
                "ik": ik,
            }

        result = self._send_joint_positions(
            [float(value) for value in q_solution[: len(_ARM_JOINTS)]],
            max(float(duration), 0.2),
            tool=tool,
        )
        result["ik_ros_success"] = bool(ik.get("ros_success"))
        result["ik_best_effort"] = not bool(ik.get("ros_success"))
        result["ik_message"] = str(ik.get("message", ""))
        return result


    def _solve_pose_ik(self, pose: Pose, *, tool: str) -> dict[str, Any]:
        request = MoveToPoseIK.Request()
        request.target_pose = pose
        try:
            result = self._call_service(
                self._ik_client,
                request,
                self._srv("move_to_pose_ik"),
            )
        except Exception as exc:
            return self._tool_error(tool, exc)

        q_solution = [float(value) for value in result.q_solution]
        return {
            "ok": bool(result.success) and bool(q_solution),
            "tool": tool,
            "ros_success": bool(result.success),
            "message": str(result.message),
            "q_solution": q_solution,
        }


    def _send_joint_positions(
        self,
        positions: list[float],
        duration: float,
        *,
        tool: str,
    ) -> dict[str, Any]:
        if not self._follow_joint_trajectory_action.wait_for_server(
            timeout_sec=self.service_timeout_sec
        ):
            return self._unavailable(
                tool,
                f"/{self.namespace}/follow_joint_trajectory",
            )

        current = self._current_joint_positions()
        if current is None:
            return {
                "ok": False,
                "tool": tool,
                "message": "No /joint_states feedback yet; cannot build safe trajectory start point.",
            }

        duration = max(float(duration), 0.2)
        target = [float(value) for value in positions[: len(_ARM_JOINTS)]]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(_ARM_JOINTS)
        goal.trajectory.points = [
            self._trajectory_point([current[name] for name in _ARM_JOINTS], 0.05),
            self._trajectory_point(target, duration),
        ]

        send_future = self._follow_joint_trajectory_action.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(
                send_future,
                self.service_timeout_sec,
                f"{tool} follow_joint_trajectory send_goal",
            )
        except Exception as exc:
            return self._tool_error(tool, exc)
        if not goal_handle.accepted:
            return {"ok": False, "tool": tool, "message": "goal rejected"}

        result_future = goal_handle.get_result_async()
        try:
            response = self._wait_future(
                result_future,
                duration + self.action_timeout_padding_sec,
                f"{tool} follow_joint_trajectory result",
            )
        except Exception as exc:
            goal_handle.cancel_goal_async()
            return self._tool_error(tool, exc)
        result = response.result
        return {
            "ok": int(result.error_code) == FollowJointTrajectory.Result.SUCCESSFUL,
            "tool": tool,
            "target_joints": dict(zip(_ARM_JOINTS, target)),
            "error_code": int(result.error_code),
            "message": str(result.error_string),
            "status": int(getattr(response, "status", 0)),
        }


    def move_joints(
        self,
        joint1: float | None = None,
        joint2: float | None = None,
        joint3: float | None = None,
        joint4: float | None = None,
        joint5: float | None = None,
        joint6: float | None = None,
        duration: float = 3.0,
        *,
        guard_motion: bool = True,
    ) -> dict[str, Any]:
        if guard_motion:
            blocked = self._motion_blocked("move_joints")
            if blocked is not None:
                return blocked

        if not self._follow_joint_trajectory_action.wait_for_server(
            timeout_sec=self.service_timeout_sec
        ):
            return self._unavailable(
                "move_joints", f"/{self.namespace}/follow_joint_trajectory"
            )

        current = self._current_joint_positions()
        if current is None:
            return {
                "ok": False,
                "tool": "move_joints",
                "message": "No /joint_states feedback yet; cannot build safe trajectory start point.",
            }

        requested = {
            "joint1": joint1,
            "joint2": joint2,
            "joint3": joint3,
            "joint4": joint4,
            "joint5": joint5,
            "joint6": joint6,
        }
        target = dict(current)
        changed = {}
        for name, value in requested.items():
            if value is not None:
                target[name] = float(value)
                changed[name] = float(value)
        if not changed:
            return {
                "ok": False,
                "tool": "move_joints",
                "message": "At least one joint target must be provided.",
            }

        duration = max(float(duration), 0.2)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(_ARM_JOINTS)
        goal.trajectory.points = [
            self._trajectory_point([current[name] for name in _ARM_JOINTS], 0.05),
            self._trajectory_point([target[name] for name in _ARM_JOINTS], duration),
        ]

        send_future = self._follow_joint_trajectory_action.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(
                send_future,
                self.service_timeout_sec,
                "follow_joint_trajectory send_goal",
            )
        except Exception as exc:
            return self._tool_error("move_joints", exc)
        if not goal_handle.accepted:
            return {"ok": False, "tool": "move_joints", "message": "goal rejected"}

        result_future = goal_handle.get_result_async()
        try:
            response = self._wait_future(
                result_future,
                duration + self.action_timeout_padding_sec,
                "follow_joint_trajectory result",
            )
        except Exception as exc:
            goal_handle.cancel_goal_async()
            return self._tool_error("move_joints", exc)
        result = response.result
        return {
            "ok": int(result.error_code) == FollowJointTrajectory.Result.SUCCESSFUL,
            "tool": "move_joints",
            "changed": changed,
            "error_code": int(result.error_code),
            "message": str(result.error_string),
            "status": int(getattr(response, "status", 0)),
        }


    def detect_blocks(self, preferred_color: str = "auto") -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._vision_parse_error is not None:
                return {
                    "ok": False,
                    "tool": "detect_blocks",
                    "message": f"vision payload parse error: {self._vision_parse_error}",
                    "age_sec": self._age(now, self._vision_payload_at),
                }
            payload = self._copy_jsonable(self._vision_payload)
            age = self._age(now, self._vision_payload_at)

        if not payload:
            return {
                "ok": False,
                "tool": "detect_blocks",
                "message": "No color block detections have been received yet.",
                "age_sec": age,
            }

        detections = payload.get("detections") or []
        color = str(preferred_color or "auto").lower()
        if color != "auto":
            detections = [
                item for item in detections if str(item.get("color", "")).lower() == color
            ]
        target = self._choose_detection(payload, color)
        return {
            "ok": True,
            "tool": "detect_blocks",
            "age_sec": age,
            "target_color": payload.get("target_color"),
            "preferred_color": color,
            "count": len(detections),
            "target": target,
            "detections": detections,
        }


    def _call_service(self, client, request, name: str):
        if not client.wait_for_service(timeout_sec=self.service_timeout_sec):
            raise TimeoutError(f"ROS service unavailable: {name}")
        future = client.call_async(request)
        return self._wait_future(future, self.service_timeout_sec, name)


    def _wait_future(self, future, timeout_sec: float, label: str):
        deadline = time.monotonic() + max(float(timeout_sec), 0.1)
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {label}")
            time.sleep(0.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f"{label} returned no result")
        return result


    def _trigger_result(self, tool: str, result) -> dict[str, Any]:
        return {
            "ok": bool(result.success),
            "tool": tool,
            "ros_success": bool(result.success),
            "message": str(result.message),
        }


    def _srv_name(self, key: str) -> str:
        return {
            "enable": self._srv("enable"),
            "disable": self._srv("disable"),
            "safe_home": self._srv("safe_home"),
            "gravity_start": self._srv("gravity_compensation/start"),
            "gravity_stop": self._srv("gravity_compensation/stop"),
            "gravity_status": self._srv("gravity_compensation/status"),
            "record_start": self._srv("mujoco/record/start"),
            "record_stop": self._srv("mujoco/record/stop"),
            "record_replay": self._srv("mujoco/record/replay"),
            "record_clear": self._srv("mujoco/record/clear"),
        }[key]


    def _motion_blocked(self, tool: str) -> dict[str, Any] | None:
        if self.motion_allowed:
            return None
        return {
            "ok": False,
            "blocked": True,
            "tool": tool,
            "message": (
                "Motion tools are locked. Restart the MCP server with "
                "--motion-mode allow after confirming you are connected to a safe "
                "simulation or intentionally controlling hardware."
            ),
        }


    def _unavailable(self, tool: str, endpoint: str) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool,
            "message": f"ROS endpoint unavailable: {endpoint}",
        }


    @staticmethod
    def _tool_error(tool: str, exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool,
            "message": str(exc),
            "error_type": exc.__class__.__name__,
        }


    def _current_joint_positions(self) -> dict[str, float] | None:
        with self._lock:
            if not self._joint_state:
                return None
            positions = self._joint_state.get("positions") or {}
            if not all(name in positions for name in _ARM_JOINTS):
                return None
            return {name: float(positions[name]) for name in _ARM_JOINTS}


    def _vision_summary_locked(self, now: float) -> dict[str, Any]:
        if self._vision_parse_error:
            return {
                "ok": False,
                "parse_error": self._vision_parse_error,
                "age_sec": self._age(now, self._vision_payload_at),
            }
        if not self._vision_payload:
            return {
                "ok": False,
                "message": "no detections yet",
                "age_sec": self._age(now, self._vision_payload_at),
            }
        detections = self._vision_payload.get("detections") or []
        return {
            "ok": True,
            "age_sec": self._age(now, self._vision_payload_at),
            "target_color": self._vision_payload.get("target_color"),
            "count": len(detections),
            "target": self._vision_payload.get("target"),
        }


    @staticmethod
    def _choose_detection(payload: dict[str, Any], preferred_color: str) -> dict[str, Any] | None:
        detections = payload.get("detections") or []
        if preferred_color != "auto":
            for item in detections:
                if str(item.get("color", "")).lower() == preferred_color:
                    return item
            return None
        target = payload.get("target")
        if isinstance(target, dict):
            return target
        return detections[0] if detections else None


    @staticmethod
    def _sleep_step(tool: str, seconds: float) -> dict[str, Any]:
        delay = max(float(seconds), 0.0)
        time.sleep(delay)
        return {"ok": True, "tool": tool, "slept_sec": round(delay, 3)}


    def _opening_mm_to_motor_rad(self, opening_mm: float) -> float:
        ratio = RosTaskClient._clamp(
            float(opening_mm) / self.settings.command_open_mm, 0.0, 1.0
        )
        return self.settings.motor_close_rad + ratio * (self.settings.motor_open_rad - self.settings.motor_close_rad)


    def _motor_rad_to_opening_mm(self, position_rad: float) -> float:
        ratio = RosTaskClient._clamp(
            (float(position_rad) - self.settings.motor_close_rad) / (self.settings.motor_open_rad - self.settings.motor_close_rad), 0.0, 1.0
        )
        return ratio * self.settings.command_open_mm


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
    def _pose_from_xyz_rpy_deg(
        x: float,
        y: float,
        z: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> Pose:
        qx, qy, qz, qw = RosTaskClient._quat_from_rpy_deg(
            roll_deg, pitch_deg, yaw_deg
        )
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose


    @staticmethod
    def _quat_from_rpy_deg(
        roll_deg: float, pitch_deg: float, yaw_deg: float
    ) -> tuple[float, float, float, float]:
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        yaw = math.radians(float(yaw_deg))
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return float(qx), float(qy), float(qz), float(qw)


    @staticmethod
    def _pose_to_dict(pose: Pose) -> dict[str, Any]:
        return {
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }


    @staticmethod
    def _trajectory_point(positions: list[float], seconds: float) -> JointTrajectoryPoint:
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        sec = int(math.floor(float(seconds)))
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = int(round((float(seconds) - sec) * 1e9))
        return point


    @staticmethod
    def _copy_jsonable(value):
        return json.loads(json.dumps(value)) if value is not None else None


    @staticmethod
    def _age(now: float, stamp: float) -> float | None:
        if stamp <= 0.0:
            return None
        return round(max(now - stamp, 0.0), 3)


    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(float(minimum), min(float(maximum), float(value)))


    def pick_color(self, *args, **kwargs):
        if not self._pick_lock.acquire(blocking=False):
            return {"ok": False, "message": "another pick task is active"}
        try:
            return self.task.pick_color(*args, **kwargs)
        finally:
            self._pick_lock.release()

    def vision_received_at(self):
        with self._lock:
            return self._vision_payload_at

    def select_target(self, color):
        msg = String()
        msg.data = color
        self._grasp_target_pub.publish(msg)

    def mark_grasp(self, color):
        if self.settings.backend == 'mujoco':
            return {'ok': True, 'message': 'simulated payload tracked from physical contacts'}
        self.select_target(color)
        return self._payload_service(self._attach_client, 'attach')

    def _payload_service(self, client, operation):
        try:
            result = self._call_service(client, Trigger.Request(), self._srv(f'planning/{operation}'))
            return {'ok': bool(result.success), 'message': result.message}
        except Exception as exc:
            return self._tool_error(f'planning.{operation}', exc)

    def move_pick_pose(self, pose, duration, *, tool):
        return self._move_to_pose_via_ik_trajectory(pose, duration, tool=tool)

    def _normalized_gripper_callback(self, msg):
        try:
            state = json.loads(msg.data)
            if not isinstance(state, dict):
                return
        except (TypeError, ValueError):
            return
        with self._lock:
            self._normalized_gripper = state
            self._normalized_gripper_at = time.monotonic()

    def gripper_snapshot(self):
        with self._lock:
            return self._copy_jsonable(self._normalized_gripper), time.monotonic() - self._normalized_gripper_at

    def wait_gripper(self, opening_mm, *, allow_contact=False):
        return wait_gripper(self.gripper_snapshot, opening_mm, self.settings,
                            allow_contact=allow_contact,
                            timeout=max(10., self.action_timeout_padding_sec))

