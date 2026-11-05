from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import threading
import time

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
import mujoco
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rebotarm_msgs.action import MoveToPose
from rebotarm_msgs.msg import ArmPositionCommand, ArmStatus
from rebotarm_msgs.srv import MoveToPoseIK
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_msgs.msg import String
from .collision_planner import CollisionPlanner, PlanningError
from .trajectory_progress import SegmentProgress, MotionWatchdog
from trajectory_msgs.msg import JointTrajectoryPoint

from .model_paths import find_default_model
from .sdk_model import SdkRobotModel
from rebotarm_description.parameters import load_mounts
from .scene import HardwareScene


_ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))


@dataclass(frozen=True)
class ArmJoint:
    name: str
    qpos_addr: int
    dof_addr: int
    lower: float
    upper: float

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.lower, self.upper))


class RsTaskServer(Node):
    """Shared bounded IK, collision planning and measured trajectory execution."""

    def __init__(self) -> None:
        super().__init__("rebotarm_rs_task_server")
        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter("backend", "mujoco")
        self.declare_parameter("hardware_scene_calibrated", False)
        self.declare_parameter("table_top_m", 0.1)
        self.declare_parameter("scene_ttl_sec", 30.0)
        self.declare_parameter("d435_collision", True)
        self.declare_parameter("arm_namespace", "rebotarm")
        self.declare_parameter("model_path", "")
        self.declare_parameter("joint_state_topic", "")
        self.declare_parameter("target_pose_topic", "")
        self.declare_parameter("tcp_body", "gripper_end")
        self.declare_parameter("tcp_offset", load_mounts()['tcp']['xyz_m'])
        self.declare_parameter("command_hz", 60.0)
        self.declare_parameter("record_hz", 30.0)
        self.declare_parameter("max_joint_speed", 1.0)
        # 位置控制的最大目标领先量。负载较大时，过小的领先量只能产生
        # 很小的 Kp 力矩，电机可能无法克服静摩擦；碰撞和 0.35 rad
        # 跟踪保护仍独立生效。
        self.declare_parameter("max_command_lead_rad", 0.20)
        # 任一未到位轴持续多久没有朝目标前进才中止；不是整条轨迹的总时长。
        self.declare_parameter("motion_stall_timeout_sec", 4.0)
        self.declare_parameter("ik_tolerance", 0.010)
        self.declare_parameter("ik_orientation_tolerance", 0.10)
        self.declare_parameter("ik_orientation_weight", 0.25)
        self.declare_parameter("ik_max_nfev", 600)
        self.declare_parameter("records_dir", "")
        # Kinematics backend: "mujoco" (default) or "sdk" (pinocchio via the
        # reBotArm_control_py SDK). Use "sdk" to make planned IK solutions match
        # the real driver / STM32.
        self.declare_parameter("kinematics_backend", "mujoco")
        self.declare_parameter("sdk_urdf", "")
        self.declare_parameter("sdk_end_effector_frame", "gripper_end")

        self.backend = str(self.get_parameter('backend').value)
        if self.backend not in ('mujoco', 'sdk', 'uart'):
            raise ValueError('unknown planning backend')
        self.hardware_scene_calibrated = bool(self.get_parameter('hardware_scene_calibrated').value)
        self.namespace = str(self.get_parameter("arm_namespace").value).strip("/")
        self.joint_state_topic = str(
            self.get_parameter("joint_state_topic").value
            or f"/{self.namespace}/joint_states"
        )
        self.target_pose_topic = str(
            self.get_parameter("target_pose_topic").value
            or f"/{self.namespace}/mujoco/target_pose"
        )
        self.command_hz = max(float(self.get_parameter("command_hz").value), 10.0)
        self.record_hz = max(float(self.get_parameter("record_hz").value), 1.0)
        self.max_joint_speed = max(
            float(self.get_parameter("max_joint_speed").value), 0.05
        )
        self.max_command_lead = float(self.get_parameter("max_command_lead_rad").value)
        if not math.isfinite(self.max_command_lead) or not 0.01 <= self.max_command_lead <= .30:
            raise ValueError("max_command_lead_rad must be between 0.01 and 0.30 rad")
        self.motion_stall_timeout = float(self.get_parameter("motion_stall_timeout_sec").value)
        if not math.isfinite(self.motion_stall_timeout) or self.motion_stall_timeout <= 0:
            raise ValueError("motion_stall_timeout_sec must be positive")
        self.ik_tolerance = max(float(self.get_parameter("ik_tolerance").value), 0.001)
        self.ik_orientation_tolerance = max(
            float(self.get_parameter("ik_orientation_tolerance").value), 0.01
        )
        self.ik_orientation_weight = max(
            float(self.get_parameter("ik_orientation_weight").value), 0.0
        )
        self.ik_max_nfev = max(int(self.get_parameter("ik_max_nfev").value), 100)
        records_dir = str(self.get_parameter("records_dir").value).strip()
        self.records_dir = (
            Path(records_dir).expanduser()
            if records_dir
            else Path.home() / ".ros" / "rebotarm_rs_records"
        )

        requested_model = str(self.get_parameter("model_path").value).strip()
        self.model_path = (
            Path(requested_model).expanduser().resolve()
            if requested_model
            else find_default_model()
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.base_qpos = self.data.qpos.copy()
        self.joints = self._build_joints()
        self.lower = np.array([joint.lower for joint in self.joints], dtype=np.float64)
        self.upper = np.array([joint.upper for joint in self.joints], dtype=np.float64)

        tcp_body_name = str(self.get_parameter("tcp_body").value)
        self.tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, tcp_body_name
        )
        if self.tcp_body_id < 0:
            raise RuntimeError(f"MuJoCo body {tcp_body_name!r} was not found")
        self.tcp_offset = np.asarray(
            self.get_parameter("tcp_offset").value, dtype=np.float64
        )
        if self.tcp_offset.shape != (3,):
            raise ValueError("tcp_offset must contain three values")

        # Optional SDK (pinocchio) kinematics backend, sharing the real
        # driver's model so planned joints transfer to hardware.
        self.sdk: SdkRobotModel | None = None
        self.kinematics_backend = (
            str(self.get_parameter("kinematics_backend").value).strip().lower()
            or "mujoco"
        )
        if self.kinematics_backend == "sdk":
            sdk_urdf = str(self.get_parameter("sdk_urdf").value).strip() or None
            self.sdk = SdkRobotModel(
                urdf_path=sdk_urdf,
                end_effector_frame=str(
                    self.get_parameter("sdk_end_effector_frame").value
                ),
                tcp_offset=self.tcp_offset,
            )
            # IK 与碰撞执行必须使用相同的关节范围，避免两套模型混用。
            for joint, bounds in zip(self.joints, self.sdk.joint_limits()):
                if not np.allclose(bounds, [joint.lower, joint.upper], rtol=0., atol=1e-7):
                    raise RuntimeError(
                        f"{joint.name}: URDF limits {bounds} differ from MJCF "
                        f"[{joint.lower}, {joint.upper}]; check robot_arm.urdf "
                        "and run scripts/sync_robot_parameters.py before restarting"
                    )
            self.get_logger().info(
                f"kinematics backend = SDK (pinocchio), urdf={self.sdk.urdf_path}"
            )
        else:
            self.get_logger().info("kinematics backend = MuJoCo + scipy")

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # 统一的下游控制接口：固定 joint1..joint6 顺序，所有后端都消费
        # 同一个绝对位置/限速消息。旧的逐关节 cmd/pos_vel 不再由规划器
        # 发布，避免仿真、UART 和未来 Servo 各自维护一套聚合逻辑。
        self.command_pub = self.create_publisher(
            ArmPositionCommand,
            f"/{self.namespace}/arm/command/position",
            reliable,
        )
        self.target_pose_pub = self.create_publisher(
            PoseStamped, self.target_pose_topic, reliable
        )

        self.planner = CollisionPlanner(self.model_path, exact_meshes=self.backend != 'mujoco')
        self._planner_lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._last_state_time = 0.0
        self._scene_time = 0.0
        self._scene_objects = {}
        self._gripper_positions = {}
        self._selected_color = ''
        self._selection_time = 0.0
        self.hardware_scene = None
        self._scene_fault = ''
        if self.backend == 'mujoco':
            self.create_subscription(String, f"/{self.namespace}/mujoco/object_states", self._scene_callback, 10, callback_group=self.callback_group)
        else:
            self.hardware_scene = HardwareScene(
                self.planner, float(self.get_parameter('table_top_m').value),
                float(self.get_parameter('scene_ttl_sec').value),
                camera_collision=bool(self.get_parameter('d435_collision').value),
            )
            self.create_subscription(String, f"/{self.namespace}/vision/color_blocks/detections", self._hardware_scene_callback, 10, callback_group=self.callback_group)
            self.create_service(Trigger, f"/{self.namespace}/planning/attach", self._attach_payload, callback_group=self.callback_group)
            self.create_service(Trigger, f"/{self.namespace}/planning/release", self._release_payload, callback_group=self.callback_group)
        self.create_subscription(String, f"/{self.namespace}/planning/grasp_target", self._target_callback, 10, callback_group=self.callback_group)
        self._state_lock = threading.RLock()
        self._ik_lock = threading.RLock()
        self._current_q = np.zeros(6, dtype=np.float64)
        self._command_q = None
        self._last_target = np.zeros(3, dtype=np.float64)
        self._arm_enabled = False
        self.create_subscription(ArmStatus, f"/{self.namespace}/arm_status", self._status_callback, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=self.callback_group)
        self._cancel = threading.Event()
        self._recording = False
        self._record_started = 0.0
        self._record_samples: list[dict[str, float]] = []

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.create_timer(
            1.0 / self.record_hz,
            self._record_timer,
            callback_group=self.callback_group,
        )
        self.create_service(
            MoveToPoseIK,
            f"/{self.namespace}/move_to_pose_ik",
            self._handle_ik,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            f"/{self.namespace}/mujoco/record/start",
            self._record_start,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            f"/{self.namespace}/mujoco/record/stop",
            self._record_stop,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            f"/{self.namespace}/mujoco/record/replay",
            self._record_replay,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            f"/{self.namespace}/mujoco/record/clear",
            self._record_clear,
            callback_group=self.callback_group,
        )

        self.create_timer(1.0 / self.command_hz, self._hold_timer, callback_group=self.callback_group)
        self.move_server = ActionServer(
            self,
            MoveToPose,
            f"/{self.namespace}/move_to_pose",
            execute_callback=self._execute_move,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self.callback_group,
        )
        self.trajectory_server = ActionServer(
            self,
            FollowJointTrajectory,
            f"/{self.namespace}/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            "RS task server ready: "
            f"namespace=/{self.namespace}, model={self.model_path}, "
            f"tcp_offset={self.tcp_offset.tolist()}"
        )

    def _status_callback(self, msg):
        with self._state_lock:
            enabled = bool(msg.enabled)
            if enabled != self._arm_enabled:
                self._command_q = None  # 重新使能必须从本次实测姿态对齐。
            self._arm_enabled = enabled

    def _hold_timer(self):
        if self.backend == 'mujoco' or (self.backend == 'sdk' and not self._arm_enabled):
            return
        try:
            with self._state_lock:
                measured = self._measured()  # 反馈过期时仍停止保持流。
                # 使能后重发锁存目标，包含空闲和重规划期间。
                # 若每周期把目标改成反馈，关节下滑时目标也会跟着下滑。
                target = self._command_q if self._arm_enabled and self._command_q is not None else measured
                self._publish_targets(target)
        except PlanningError:
            return

    def _build_joints(self) -> list[ArmJoint]:
        result = []
        for name in _ARM_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"MuJoCo joint {name!r} was not found")
            result.append(
                ArmJoint(
                    name=name,
                    qpos_addr=int(self.model.jnt_qposadr[joint_id]),
                    dof_addr=int(self.model.jnt_dofadr[joint_id]),
                    lower=float(self.model.jnt_range[joint_id, 0]),
                    upper=float(self.model.jnt_range[joint_id, 1]),
                )
            )
        return result

    def _joint_state_callback(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        if not all(name in values and math.isfinite(float(values[name])) for name in _ARM_JOINTS):
            return
        with self._state_lock:
            for name in ('gripper_joint1', 'gripper_joint2'):
                if name in values and math.isfinite(float(values[name])):
                    self._gripper_positions[name] = float(values[name])
            self._last_state_time = time.monotonic()
            for index, name in enumerate(_ARM_JOINTS):
                value = values.get(name)
                if value is not None and math.isfinite(float(value)):
                    self._current_q[index] = self.joints[index].clamp(float(value))

    def _target_callback(self, msg):
        color = msg.data.strip().lower()
        self._selected_color = color if color in ('red', 'blue', 'yellow') else ''
        self._selection_time = time.monotonic()

    def _scene_callback(self, msg):
        try:
            objects = json.loads(msg.data)['objects']
            snapshot = {}
            for item in objects:
                body = self.planner.model.body(item['name'])
                joint = int(self.planner.model.body_jntadr[body.id])
                if joint < 0 or self.planner.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                adr = int(self.planner.model.jnt_qposadr[joint])
                xyz = np.asarray(item['position'], dtype=float)
                quat = np.asarray(item['quaternion'], dtype=float)
                if xyz.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(np.r_[xyz, quat])) or np.linalg.norm(quat) < 1e-6:
                    return
                snapshot[adr] = np.r_[xyz, quat/np.linalg.norm(quat)]
            with self._state_lock:
                self._scene_objects = snapshot
                self._scene_time = time.monotonic()
        except (KeyError, ValueError, TypeError):
            return

    def _hardware_scene_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            with self._planner_lock:
                self.hardware_scene.update(payload)
                self._scene_fault = ''
        except PlanningError as exc:
            self._scene_fault = str(exc)
            self.get_logger().error(self._scene_fault)
        except (ValueError, TypeError, KeyError):
            return

    def _attach_payload(self, _request, response):
        try:
            with self._planner_lock:
                self.hardware_scene.attach(self._selected_color, self._measured())
            response.success, response.message = True, 'measured payload included in collision planning'
        except (PlanningError, ValueError, KeyError) as exc:
            response.success, response.message = False, str(exc)
        return response

    def _release_payload(self, _request, response):
        try:
            with self._planner_lock:
                self.hardware_scene.attach('release', self._measured())
            response.success, response.message = True, 'payload released from planning scene'
        except PlanningError as exc:
            response.success, response.message = False, str(exc)
        return response

    def _prepare_collision_scene(self):
        if self.backend != 'mujoco':
            if self._scene_fault:
                raise PlanningError(self._scene_fault)
            if not self.hardware_scene_calibrated:
                raise PlanningError('hardware fixture scene has not been calibrated')
            with self._state_lock:
                self.hardware_scene.prepare(self._selected_color, self._measured(), dict(self._gripper_positions))
            return
        with self._state_lock:
            if time.monotonic()-self._scene_time > 1.0:
                raise PlanningError('simulation collision scene is stale')
            names = {'red': 'red_cube', 'blue': 'blue_block', 'yellow': 'yellow_cylinder'}
            selected = names.get(self._selected_color) if time.monotonic()-self._selection_time < 15.0 else None
            self.planner.allowed_object = self.planner.model.body(selected).id if selected else None
            for adr, pose in self._scene_objects.items():
                self.planner.data.qpos[adr:adr+7] = pose
            for name, position in self._gripper_positions.items():
                self.planner.data.qpos[self.planner.model.joint(name).qposadr[0]] = position
            measured = self._measured()
        self.planner.capture_grasp(measured)

    def _measured(self):
        with self._state_lock:
            if time.monotonic()-self._last_state_time > .5:
                raise PlanningError('joint feedback is missing or stale')
            return self._current_q.copy()

    def _accept_goal(self, _request):
        if self.backend != 'mujoco' and not self._arm_enabled:
            self.get_logger().warning('Trajectory rejected: hardware is disabled; check /rebotarm/uart/status')
            return GoalResponse.REJECT
        return GoalResponse.REJECT if self._motion_lock.locked() else GoalResponse.ACCEPT

    def _accept_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _handle_ik(self, request, response):
        target = self._pose_xyz(request.target_pose)
        target_mat = self._pose_matrix(request.target_pose)
        solution, pos_error, orientation_error = self._solve_ik(
            target, self._current_positions(), target_mat
        )
        response.q_solution = solution.tolist()
        response.success = bool(
            pos_error <= self.ik_tolerance
            and (
                target_mat is None
                or orientation_error <= self.ik_orientation_tolerance
            )
        )
        response.message = (
            f"RS IK {'success' if response.success else 'best effort'}, "
            f"position={pos_error * 1000.0:.1f} mm, orientation={orientation_error:.4f}"
        )
        return response

    def _execute_move(self, goal_handle):
        request = goal_handle.request
        target = self._pose_xyz(request.target_pose)
        target_mat = self._pose_matrix(request.target_pose)
        start = self._current_positions()
        solution, pos_error, orientation_error = self._solve_ik(target, start, target_mat)
        result = MoveToPose.Result()
        if pos_error > self.ik_tolerance or (
            target_mat is not None
            and orientation_error > self.ik_orientation_tolerance
        ):
            goal_handle.abort()
            result.success = False
            result.message = (
                f"RS IK failed: position={pos_error * 1000.0:.1f} mm, "
                f"orientation={orientation_error:.4f}"
            )
            result.final_pose = self._fk_pose_msg(start)
            return result

        duration = self._safe_duration(start, solution, float(request.duration))
        try:
            ok = self._run_path([start, solution], [0.0, duration], goal_handle=goal_handle, move_feedback=True)
        except PlanningError as exc:
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
            return result
        result.final_pose = self._fk_pose_msg(self._current_positions())
        if not ok:
            goal_handle.canceled()
            result.success = False
            result.message = "RS Cartesian motion canceled"
            return result
        goal_handle.succeed()
        result.success = True
        result.message = f"RS Cartesian motion complete, IK error={pos_error * 1000.0:.1f} mm"
        return result

    def _execute_trajectory(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        try:
            positions, times = self._expand_trajectory(
                list(trajectory.joint_names), list(trajectory.points)
            )
        except ValueError as exc:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
            return result

        try:
            ok = self._run_path(positions, times, goal_handle=goal_handle)
        except PlanningError as exc:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = str(exc)
            return result
        if not ok:
            goal_handle.canceled()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory canceled"
            return result
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "RS trajectory complete"
        return result

    def _expand_trajectory(
        self, names: list[str], points: list[JointTrajectoryPoint]
    ) -> tuple[list[np.ndarray], list[float]]:
        if not names or not points:
            raise ValueError("trajectory requires joint_names and points")
        if len(names) != len(set(names)):
            raise ValueError("trajectory joint_names contain duplicates")
        if not set(names).issubset(_ARM_JOINTS):
            raise ValueError("trajectory contains unknown RS joints")

        path = [self._current_positions()]
        times = [0.0]
        for point_index, point in enumerate(points):
            if len(point.positions) != len(names):
                raise ValueError("trajectory positions length does not match joint_names")
            next_q = path[-1].copy()
            for name, value in zip(names, point.positions):
                index = _ARM_JOINTS.index(name)
                value = float(value)
                joint = self.joints[index]
                if not math.isfinite(value) or not joint.lower - 0.001 <= value <= joint.upper + 0.001:
                    raise ValueError(
                        f'trajectory point {point_index}: {name} position={value:.9g} rad, '
                        f'allowed=[{joint.lower:.9g}, {joint.upper:.9g}] rad; '
                        'check MoveIt and execution model consistency'
                    )
                next_q[index] = self.joints[index].clamp(value)
            requested = float(point.time_from_start.sec) + float(
                point.time_from_start.nanosec
            ) / 1e9
            minimum = times[-1] + self._safe_duration(path[-1], next_q, 0.0)
            path.append(next_q)
            times.append(max(requested, minimum))
        return path, times

    def _run_path(self, *args, **kwargs):
        if not self._motion_lock.acquire(blocking=False):
            raise PlanningError('another motion is active')
        started = time.monotonic()
        ok = False
        try:
            with self._state_lock:
                self._command_q = self._measured()
            ok = self._run_path_impl(*args, **kwargs)
            self.get_logger().info(
                f'trajectory {"complete" if ok else "canceled"}: '
                f'elapsed={time.monotonic()-started:.3f}s; {self._tracking_summary()}'
            )
            return ok
        except PlanningError as exc:
            self.get_logger().error(f'trajectory aborted: {exc}; {self._tracking_summary()}')
            raise
        finally:
            try:
                # 成功后继续保持已校验的终点，不把尚有误差的反馈重设为目标。
                # 取消或失败才撤回运动目标，防止故障后继续追赶。
                if not ok:
                    with self._state_lock:
                        measured = self._current_q.copy()
                    self._publish_targets(measured)
            finally:
                self._motion_lock.release()

    def _tracking_summary(self):
        """记录主机目标和实测角度；它不能代替 STM32 的指令确认。"""
        with self._state_lock:
            measured = self._current_q.copy()
            target = self._command_q.copy() if self._command_q is not None else measured
        return (f'measured_rad={np.round(measured, 5).tolist()}, '
                f'target_rad={np.round(target, 5).tolist()}, '
                f'max_error_rad={np.max(np.abs(target-measured)):.5f}')

    def _run_path_impl(
        self,
        path: list[np.ndarray],
        times: list[float],
        *,
        goal_handle=None,
        move_feedback: bool = False,
    ) -> bool:
        if len(path) < 2 or len(path) != len(times):
            return True
        self._cancel.clear()
        period = 1.0 / self.command_hz
        checked_path = [self._measured()]
        if self.backend != 'mujoco':
            self.get_logger().info(
                f'trajectory requested: points={len(path)}, duration={times[-1]:.3f}s, '
                f'start_rad={np.round(checked_path[0], 5).tolist()}, '
                f'goal_rad={np.round(path[-1], 5).tolist()}'
            )
        checked_times = [0.0]
        for i in range(1, len(path)):
            with self._planner_lock:
                self._prepare_collision_scene()
                segment_path = self.planner.plan(checked_path[-1], path[i])
            requested = max(0., times[i]-times[i-1])
            for endpoint in segment_path[1:]:
                duration = self._safe_duration(checked_path[-1], endpoint, requested/(len(segment_path)-1))
                checked_times.append(checked_times[-1]+duration)
                checked_path.append(endpoint)
        path, times = checked_path, checked_times
        started = time.monotonic()
        next_report = started
        next_feedback = started
        total = max(times[-1], period)
        watchdog = MotionWatchdog(self._measured(), started,
                                  timeout=getattr(self, 'motion_stall_timeout', 4.))

        def report(measured, target, nominal_elapsed, now, phase):
            """推进和终点等待阶段都发布实际误差；到终点附近不等于已成功。"""
            nonlocal next_report, next_feedback
            if self.backend != 'mujoco' and now >= next_report:
                self.get_logger().info(
                    f'trajectory {phase}: elapsed={now-started:.3f}s; {self._tracking_summary()}'
                )
                next_report = now + 1.0
            if move_feedback and goal_handle is not None:
                feedback = MoveToPose.Feedback()
                feedback.current_pose = self._fk_pose_msg(measured)
                feedback.progress = float(np.clip(nominal_elapsed / total, 0.0, 1.0))
                feedback.time_elapsed = now-started
                goal_handle.publish_feedback(feedback)
            elif goal_handle is not None and now >= next_feedback:
                feedback = FollowJointTrajectory.Feedback()
                feedback.header.stamp = self.get_clock().now().to_msg()
                feedback.joint_names = list(_ARM_JOINTS)
                feedback.desired.positions = target.tolist()
                feedback.actual.positions = measured.tolist()
                feedback.error.positions = (target-measured).tolist()
                elapsed = now-started
                feedback.desired.time_from_start.sec = int(elapsed)
                feedback.desired.time_from_start.nanosec = int((elapsed-int(elapsed))*1e9)
                feedback.actual.time_from_start = feedback.desired.time_from_start
                goal_handle.publish_feedback(feedback)
                next_feedback = now + .1

        for segment in range(1, len(path)):
            q0, q1 = path[segment - 1], path[segment]
            duration = max(times[segment]-times[segment-1], period)
            progress = SegmentProgress(
                q0, q1, duration,
                max_lead=getattr(self, 'max_command_lead', .08),
            )
            previous_tick = time.monotonic()
            while True:
                if self._cancel.is_set() or (
                    goal_handle is not None and goal_handle.is_cancel_requested
                ):
                    return False
                now = time.monotonic()
                if self.backend != 'mujoco' and not self._arm_enabled:
                    raise PlanningError('hardware disabled during motion')
                measured = self._measured()
                previous_target = progress.at(progress.fraction)
                tracking_error = np.abs(measured-previous_target)
                worst_joint = int(np.argmax(tracking_error))
                if tracking_error[worst_joint] > .35:
                    raise PlanningError(
                        "joint tracking error exceeds 0.35 rad: "
                        f"{_ARM_JOINTS[worst_joint]} "
                        f"measured={measured[worst_joint]:.4f}, "
                        f"commanded={previous_target[worst_joint]:.4f}, "
                        f"error={tracking_error[worst_joint]:.4f}"
                    )
                try:
                    # 界面或计算延迟不应造成目标跳变；跟随较慢时延长执行时间。
                    q = progress.advance(measured, min(now-previous_tick, period))
                    watchdog.check(measured, q, now)
                except RuntimeError as exc:
                    raise PlanningError(str(exc)) from exc
                previous_tick = now
                with self._planner_lock:
                    self._prepare_collision_scene()
                    if not self.planner.edge(measured, q):
                        raise PlanningError(f'path blocked during execution: {self.planner.last_collision}')
                self._publish_targets(q)
                report(measured, q, times[segment-1]+progress.fraction*duration, now, 'progress')
                if progress.fraction >= 1.0:
                    break
                time.sleep(period)
        # 指令已到终点后继续等待实物；缓慢接近不会因固定 4 秒期限被打断。
        # 同一逐轴看门狗、反馈超时、碰撞和大跟踪误差检查仍持续生效。
        stable_since = None
        while True:
            if self._cancel.is_set() or (goal_handle is not None and goal_handle.is_cancel_requested):
                return False
            if self.backend != 'mujoco' and not self._arm_enabled:
                raise PlanningError('hardware disabled while settling at goal')
            measured = self._measured()
            now = time.monotonic()
            if np.max(np.abs(measured-path[-1])) > .35:
                raise PlanningError('joint tracking error exceeds 0.35 rad while settling at goal')
            try:
                watchdog.check(measured, path[-1], now)
            except RuntimeError as exc:
                raise PlanningError(str(exc)) from exc
            with self._planner_lock:
                self._prepare_collision_scene()
                if not self.planner.edge(measured, path[-1]):
                    raise PlanningError('collision while settling at goal')
            self._publish_targets(path[-1])
            report(measured, path[-1], total, now, 'settling')
            if np.max(np.abs(measured-path[-1])) <= watchdog.tolerance:
                if stable_since is None:
                    stable_since = now
                if now-stable_since >= .2:
                    return True
            else:
                stable_since = None
            time.sleep(period)

    def _publish_targets(self, q: np.ndarray) -> None:
        # 心跳和轨迹执行共用此锁，避免旧的保持目标覆盖刚更新的轨迹点。
        with self._state_lock:
            self._command_q = np.asarray(q, dtype=float).copy()
            msg = ArmPositionCommand()
            msg.stamp = self.get_clock().now().to_msg()
            msg.positions = [self.joints[index].clamp(float(q[index]))
                             for index in range(6)]
            msg.velocity_limits = [float(self.max_joint_speed)] * 6
            self.command_pub.publish(msg)

    def _solve_ik(
        self,
        target: np.ndarray,
        start: np.ndarray,
        target_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, float, float]:
        with self._ik_lock:
            seeds = self._ik_seeds(start, target)
            best: tuple[float, np.ndarray, float, float] | None = None
            for seed in seeds:
                try:
                    solved = least_squares(
                        self._ik_residual,
                        seed,
                        args=(target, target_mat),
                        bounds=(self.lower, self.upper),
                        max_nfev=self.ik_max_nfev,
                        xtol=1e-9,
                        ftol=1e-9,
                        gtol=1e-9,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                q = np.clip(solved.x, self.lower, self.upper)
                position, rotation = self._fk_pose(q)
                pos_error = float(np.linalg.norm(target - position))
                orientation_error = (
                    0.0
                    if target_mat is None
                    else float(np.linalg.norm(self._orientation_error(rotation, target_mat)))
                )
                distance = float(np.linalg.norm((q - start) / np.maximum(self.upper - self.lower, 0.1)))
                score = pos_error + 0.25 * orientation_error + 0.002 * distance
                candidate = (score, q.copy(), pos_error, orientation_error)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if pos_error <= self.ik_tolerance and (
                    target_mat is None
                    or orientation_error <= self.ik_orientation_tolerance
                ) and distance < 0.8:
                    break
            if best is None:
                return start.copy(), float("inf"), float("inf")
            return best[1], best[2], best[3]

    def _ik_seeds(self, start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        seeds = [
            start.copy(),
            np.array([0.0, 1.6, 1.2, -0.8, 0.0, 0.0]),
            np.array([0.2, 1.2, 0.5, 0.0, 0.5, -2.5]),
            np.array([-0.2, 1.2, 0.5, 0.0, -0.5, 2.5]),
            np.array([0.0, 2.0, 1.7, -1.3, 0.0, 0.0]),
        ]
        seed_value = int(abs(float(target[0] * 997 + target[1] * 577 + target[2] * 313)) * 1e6)
        rng = np.random.default_rng(seed_value)
        for _ in range(3):
            seeds.append(self.lower + rng.random(6) * (self.upper - self.lower))
        epsilon = 1e-5
        return [np.clip(seed, self.lower + epsilon, self.upper - epsilon) for seed in seeds]

    def _ik_residual(
        self, q: np.ndarray, target: np.ndarray, target_mat: np.ndarray | None
    ) -> np.ndarray:
        position, rotation = self._fk_pose(q)
        position_error = target - position
        if target_mat is None or self.ik_orientation_weight <= 0.0:
            return position_error
        return np.concatenate(
            [
                position_error,
                self.ik_orientation_weight
                * self._orientation_error(rotation, target_mat),
            ]
        )

    def _fk_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._ik_lock:
            return self._fk_pose_locked(q)

    def _fk_pose_locked(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # SDK backend: same FK as the real driver (pinocchio). The MuJoCo model
        # is only used as the simulation plant, not for kinematics here.
        if self.sdk is not None:
            return self.sdk.fk(q)
        self.data.qpos[:] = self.base_qpos
        self.data.qvel[:] = 0.0
        for index, joint in enumerate(self.joints):
            self.data.qpos[joint.qpos_addr] = joint.clamp(float(q[index]))
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.tcp_body_id].reshape(3, 3).copy()
        position = self.data.xpos[self.tcp_body_id].copy() + rotation @ self.tcp_offset
        return position, rotation

    def _fk_pose_msg(self, q: np.ndarray) -> Pose:
        position, rotation = self._fk_pose(q)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rotation.reshape(9))
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position.tolist()
        pose.orientation.w = float(quat[0])
        pose.orientation.x = float(quat[1])
        pose.orientation.y = float(quat[2])
        pose.orientation.z = float(quat[3])
        return pose

    def _record_timer(self) -> None:
        with self._state_lock:
            if not self._recording:
                return
            sample = {"t": time.monotonic() - self._record_started}
            sample.update({name: float(value) for name, value in zip(_ARM_JOINTS, self._current_q)})
            self._record_samples.append(sample)

    def _record_start(self, _request, response):
        with self._state_lock:
            self._record_samples = []
            self._record_started = time.monotonic()
            self._recording = True
        response.success = True
        response.message = "RS task recording started"
        return response

    def _record_stop(self, _request, response):
        with self._state_lock:
            self._recording = False
            samples = list(self._record_samples)
        if not samples:
            response.success = False
            response.message = "No RS task samples recorded"
            return response
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self.records_dir / f"rs_record_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["t", *_ARM_JOINTS])
            writer.writeheader()
            writer.writerows(samples)
        response.success = True
        response.message = f"Saved {len(samples)} RS samples to {path}"
        return response

    def _record_replay(self, _request, response):
        with self._state_lock:
            samples = list(self._record_samples)
            self._recording = False
        if len(samples) < 2:
            response.success = False
            response.message = "Need at least two RS samples to replay"
            return response
        path = [np.array([sample[name] for name in _ARM_JOINTS]) for sample in samples]
        times = [float(sample["t"]) for sample in samples]
        if self._motion_lock.locked():
            response.success = False
            response.message = "another motion is active"
            return response
        def replay():
            try:
                self._run_path(path, times)
            except PlanningError as exc:
                self.get_logger().error(f"Replay stopped: {exc}")
        threading.Thread(target=replay, daemon=True).start()
        response.success = True
        response.message = f"Replaying {len(samples)} RS samples"
        return response

    def _record_clear(self, _request, response):
        self._cancel.set()
        with self._state_lock:
            self._recording = False
            self._record_samples = []
        response.success = True
        response.message = "RS task recording cleared"
        return response

    def _current_positions(self) -> np.ndarray:
        with self._state_lock:
            return self._current_q.copy()

    def _safe_duration(self, start: np.ndarray, target: np.ndarray, requested: float) -> float:
        required = 1.5 * float(np.max(np.abs(target - start))) / self.max_joint_speed
        # MoveIt 的密集采样点可不足 0.2 s；逐点加这个下限会拉长整条轨迹。
        # 保留平滑插值的限速要求，最短时间仅取一个控制周期。
        return max(float(requested), required, 1.0 / self.command_hz)

    @staticmethod
    def _pose_xyz(pose: Pose) -> np.ndarray:
        return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    @staticmethod
    def _pose_matrix(pose: Pose) -> np.ndarray | None:
        quat = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return None
        quat /= norm
        if np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-5):
            return None
        matrix = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, quat)
        return matrix.reshape(3, 3)

    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        # The former cross-product approximation becomes exactly zero for a
        # 180-degree error.  That made an upside-down gripper look perfect to
        # the solver.  A rotation vector preserves both the error axis and the
        # full angle all the way to pi radians.
        relative = target @ current.T
        return Rotation.from_matrix(relative).as_rotvec()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RsTaskServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
