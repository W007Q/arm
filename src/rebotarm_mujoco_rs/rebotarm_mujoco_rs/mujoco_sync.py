from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rebotarm_msgs.msg import ArmPositionCommand
from rebotarm_msgs.srv import GripperCommand, SetGripper
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .motor_model import MotorModel, load_motor_model
from .joint_control import pd_bias_torque, limit_arm_torque
from rebotarm_description.parameters import arm_gains, description_path, hardware_profile, joint_limits, load_hardware
from rebotarm_description.command_interface import validate_position_command


_ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
_GRIPPER_MOTOR_OPEN_RAD = float(hardware_profile(load_hardware())['gripper']['position_limits']['open'])
_GRIPPER_MOTOR_CLOSE_RAD = float(hardware_profile(load_hardware())['gripper']['position_limits']['close'])
_RS_ROS_VISUAL_OPEN_M = 0.045
_LEGACY_VISUAL_OPEN_M = 0.0285
# Max finger travel of the current robot model. The gripper is driven by the
# master joint7 (0 = closed, URDF finger travel = open); an <equality> constraint syncs the
# two fingers: gripper_joint1 = -joint7, gripper_joint2 = +joint7.
_MUJOCO_GRIPPER_OPEN_M = joint_limits()['gripper_joint2']['upper']
_GRASP_SCENE_RELATIVE = Path("src/rebotarm_description/mjcf/grasp_scene.xml")
_DEFAULT_OBJECTS = ("red_cube", "blue_block", "yellow_cylinder")


def find_default_model() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        grasp_scene = parent / _GRASP_SCENE_RELATIVE
        if grasp_scene.is_file():
            return grasp_scene
    raise FileNotFoundError(
        "Current rebotarm_description grasp scene was not found; "
        "pass model_path explicitly if installed outside this workspace."
    )


class RsMujocoSync(Node):
    """Run RS MuJoCo dynamics and synchronize them with ROS joint targets."""

    def __init__(self) -> None:
        super().__init__("rebotarm_rs_mujoco")

        self.declare_parameter("arm_namespace", "rebotarm")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("model_path", "")
        # Physics is the useful default for controller and contact validation.
        # Kinematic mode remains available for fast ROS/MoveIt integration tests.
        self.declare_parameter("simulation_mode", "physics")
        self.declare_parameter("update_rate", 250.0)
        self.declare_parameter("smoothing_alpha", 1.0)
        self.declare_parameter("stale_timeout", 1.0)
        self.declare_parameter("use_viewer", False)
        self.declare_parameter("viewer_rate", 30.0)
        self.declare_parameter("object_names", list(_DEFAULT_OBJECTS))
        self.declare_parameter("object_publish_rate", 30.0)
        # Match the RS hardware configuration's MIT gains by default. These
        # values remain launch parameters for model-specific calibration.
        gains = arm_gains()
        model_limits = joint_limits()
        self.declare_parameter("arm_kp", gains['kp'])
        self.declare_parameter("arm_kd", gains['kd'])
        self.declare_parameter("arm_tau_limit", [model_limits[name]['effort'] for name in _ARM_JOINTS])
        # Scales qfrc_bias (gravity and velocity bias); 1 preserves normal control.
        self.declare_parameter("gravity_bias_scale", 1.0)
        self.declare_parameter(
            "arm_target_speed", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.declare_parameter("motor_specs", "")
        self.declare_parameter("motor_model_enabled", True)
        self.declare_parameter("compliance_enabled", False)
        self.declare_parameter("compliance_scale", 0.5)
        self.declare_parameter("contact_torque_threshold", 1.0)
        self.declare_parameter("authority_mode", False)
        self.declare_parameter("joint_states_topic", "")
        self.declare_parameter("subscribe_commands", False)
        self.declare_parameter("legacy_joint_command_topics", False)
        # Default: start at the exported CAD pose (all joints zero).
        self.declare_parameter(
            "initial_positions", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        # The current CAD fingers have very small inertia, so the old
        # high-gain PD (300/120) is numerically unstable at the 2 ms physics
        # step. Lower gains plus a little reflected armature keep them stable.
        self.declare_parameter("gripper_kp", 80.0)
        self.declare_parameter("gripper_kd", 10.0)
        self.declare_parameter("gripper_tau_limit", 150.0)
        self.declare_parameter("gripper_force_limit", 0.0)
        # target_gripper is a MuJoCo slide in metres. 2 rad/s at the real
        # 0..5 rad motor range maps to 0.02 m/s over the 0..0.05 m slide.
        self.declare_parameter("gripper_target_speed", 0.02)
        self.declare_parameter("provide_gripper_services", False)
        self.declare_parameter("control_mode", "joint")
        self.declare_parameter("cartesian_target_topic", "")
        self.declare_parameter("cartesian_kp", 200.0)
        self.declare_parameter("cartesian_kd", 20.0)
        self.declare_parameter("cartesian_rot_kp", 20.0)
        self.declare_parameter("cartesian_rot_kd", 2.0)
        self.declare_parameter("pingpong_enabled", False)
        self.declare_parameter("pingpong_state_rate", 120.0)

        namespace = str(self.get_parameter("arm_namespace").value).strip("/")
        input_topic = str(self.get_parameter("input_topic").value).strip()
        output_topic = str(self.get_parameter("output_topic").value).strip()
        authority_mode = bool(self.get_parameter("authority_mode").value)
        if not input_topic:
            # In authority mode the node publishes /joint_states itself, so do
            # not subscribe to that same topic (would create a feedback loop).
            input_topic = (
                f"/{namespace}/mujoco/input"
                if authority_mode
                else f"/{namespace}/joint_states"
            )
        output_topic = output_topic or f"/{namespace}/mujoco/joint_states"

        requested_path = str(self.get_parameter("model_path").value).strip()
        self.model_path = (
            Path(requested_path).expanduser().resolve()
            if requested_path
            else find_default_model()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        self.simulation_mode = str(
            self.get_parameter("simulation_mode").value
        ).strip().lower()
        if self.simulation_mode not in ("kinematic", "physics"):
            raise ValueError("simulation_mode must be 'kinematic' or 'physics'")

        self.update_rate = max(float(self.get_parameter("update_rate").value), 1.0)
        self.smoothing_alpha = float(
            np.clip(self.get_parameter("smoothing_alpha").value, 0.01, 1.0)
        )
        self.stale_timeout = max(
            float(self.get_parameter("stale_timeout").value), 0.0
        )
        self.arm_kp = self._vector_parameter("arm_kp")
        self.arm_kd = self._vector_parameter("arm_kd")
        self.arm_tau_limit = self._vector_parameter("arm_tau_limit")
        self.gravity_bias_scale = float(self.get_parameter("gravity_bias_scale").value)
        if not np.isfinite(self.gravity_bias_scale) or self.gravity_bias_scale < 0:
            raise ValueError("gravity_bias_scale must be finite and non-negative")
        self.arm_target_speed = self._vector_parameter("arm_target_speed")
        self.motor_model: MotorModel | None = None
        if bool(self.get_parameter("motor_model_enabled").value):
            specs_path = str(self.get_parameter("motor_specs").value).strip()
            specs_path = specs_path or self._default_motor_specs_path()
            if specs_path and Path(specs_path).is_file():
                self.motor_model = load_motor_model(specs_path)
                self.get_logger().info(f"motor model loaded: {specs_path}")
            else:
                self.get_logger().warn(
                    f"motor specs not found ({specs_path!r}); using fixed limits"
                )
        self.gripper_kp = float(self.get_parameter("gripper_kp").value)
        self.gripper_kd = float(self.get_parameter("gripper_kd").value)
        self.gripper_tau_limit = float(self.get_parameter("gripper_tau_limit").value)
        self.gripper_force_limit = float(self.get_parameter("gripper_force_limit").value)
        self._command_gripper_force_limit = self.gripper_force_limit
        self.gripper_target_speed = max(
            float(self.get_parameter("gripper_target_speed").value), 0.01
        )
        self.compliance_enabled = bool(self.get_parameter("compliance_enabled").value)
        self.compliance_scale = float(
            np.clip(self.get_parameter("compliance_scale").value, 0.0, 1.0)
        )
        self.contact_torque_threshold = float(
            self.get_parameter("contact_torque_threshold").value
        )
        self.authority_mode = authority_mode
        # Authority mode is a standalone simulated controller. It must consume
        # the same low-level position/velocity commands as the real driver;
        # requiring a second opt-in here would silently produce a non-moving sim.
        self.subscribe_commands = bool(
            self.get_parameter("subscribe_commands").value
        ) or authority_mode
        self.provide_gripper_services = bool(
            self.get_parameter("provide_gripper_services").value
        ) or authority_mode
        self.control_mode = (
            str(self.get_parameter("control_mode").value).strip().lower() or "joint"
        )
        self.cartesian_kp = float(self.get_parameter("cartesian_kp").value)
        self.cartesian_kd = float(self.get_parameter("cartesian_kd").value)
        self.cartesian_rot_kp = float(self.get_parameter("cartesian_rot_kp").value)
        self.cartesian_rot_kd = float(self.get_parameter("cartesian_rot_kd").value)
        self.cartesian_target: PoseStamped | None = None
        self._last_arm_tau_ext = np.zeros(len(_ARM_JOINTS))
        self._last_gripper_force = 0.0

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.pingpong_enabled = bool(self.get_parameter("pingpong_enabled").value)
        self.pingpong_paddle_body = -1
        self.pingpong_ball_body = -1
        self.pingpong_ball_qpos = -1
        if self.pingpong_enabled:
            # The grasp scene intentionally has no ping-pong geometry. Keep
            # the controller code available, but fail closed on this scene.
            missing = any(
                mujoco.mj_name2id(self.model, object_type, name) < 0
                for object_type, name in (
                    (mujoco.mjtObj.mjOBJ_BODY, "pingpong_paddle"),
                    (mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball"),
                    (mujoco.mjtObj.mjOBJ_JOINT, "pingpong_ball_freejoint"),
                )
            )
            if missing:
                self.get_logger().warning(
                    "pingpong_enabled requested, but this grasp model has no ping-pong bodies; disabled"
                )
                self.pingpong_enabled = False
            else:
                self.pingpong_paddle_body = self._required_id(
                    mujoco.mjtObj.mjOBJ_BODY, "pingpong_paddle"
                )
                self.pingpong_ball_body = self._required_id(
                    mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball"
                )
                ball_joint = self._required_id(
                    mujoco.mjtObj.mjOBJ_JOINT, "pingpong_ball_freejoint"
                )
                self.pingpong_ball_qpos = int(self.model.jnt_qposadr[ball_joint])
        mujoco.mj_forward(self.model, self.data)
        self.arm_joint_ids = np.array(
            [self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in _ARM_JOINTS]
        )
        self.arm_qpos_addrs = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_addrs = self.model.jnt_dofadr[self.arm_joint_ids]
        self._cart_jacp = np.zeros((3, self.model.nv))
        self._cart_jacr = np.zeros((3, self.model.nv))
        if self.motor_model is not None:
            self._apply_motor_model_to_model()
        self.arm_actuator_ids = np.array(
            [
                self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
                for name in _ARM_JOINTS
            ]
        )
        # Gripper keeps the original single-master design: joint7 is driven by
        # joint7_motor and an <equality> syncs the fingers
        # (gripper_joint1 = -joint7, gripper_joint2 = +joint7). The finger
        # joints are only read for publishing.
        self.left_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1"
        )
        self.right_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2"
        )
        self.left_qpos_addr = int(self.model.jnt_qposadr[self.left_joint_id])
        self.right_qpos_addr = int(self.model.jnt_qposadr[self.right_joint_id])
        self.left_dof_addr = int(self.model.jnt_dofadr[self.left_joint_id])
        self.right_dof_addr = int(self.model.jnt_dofadr[self.right_joint_id])
        self.master_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, "joint7"
        )
        self.master_qpos_addr = int(self.model.jnt_qposadr[self.master_joint_id])
        self.master_dof_addr = int(self.model.jnt_dofadr[self.master_joint_id])
        self.master_actuator_id = self._required_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_motor"
        )
        self._tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_end"
        )
        if self._tcp_body_id < 0:
            self._tcp_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link6"
            )
        self.object_names = [
            str(name) for name in self.get_parameter("object_names").value
        ]
        self.object_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.object_names
        }
        self.object_body_ids = {
            name: body_id
            for name, body_id in self.object_body_ids.items()
            if body_id >= 0
        }
        self.finger_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("gripper_left", "gripper_right")
        }
        self.finger_body_ids = {
            name: body_id for name, body_id in self.finger_body_ids.items() if body_id >= 0
        }
        self._finger_names_by_id = {
            body_id: name for name, body_id in self.finger_body_ids.items()
        }
        self.object_publish_period = 1.0 / max(
            float(self.get_parameter("object_publish_rate").value), 1.0
        )

        # The CAD zero pose folds the arm down (gripper below the base), which
        # would poke through a table the base sits on. Start from a raised home
        # pose instead. Values outside the model limits are ignored.
        initial = np.asarray(
            self.get_parameter("initial_positions").value, dtype=np.float64
        )
        if initial.shape == (len(_ARM_JOINTS),) and np.all(np.isfinite(initial)):
            lower = self.model.jnt_range[self.arm_joint_ids, 0]
            upper = self.model.jnt_range[self.arm_joint_ids, 1]
            initial = np.clip(initial, lower, upper)
            self.data.qpos[self.arm_qpos_addrs] = initial
        self.target_arm = self.data.qpos[self.arm_qpos_addrs].copy()
        self.target_gripper = 0.0
        # The real POS_VEL path cannot teleport to a new position target. Keep
        # a separate, rate-limited target to model that actuator-side behavior.
        self._applied_arm_target = self.target_arm.copy()
        self._applied_gripper_target = float(self.target_gripper)
        self._arm_command_speed = self.arm_target_speed.copy()
        # In authority mode the node is the plant: run immediately and publish
        # joint states so planners can read the initial state.
        self.last_input_time = time.monotonic() if self.authority_mode else 0.0
        self.last_publish_time = 0.0
        self.last_object_publish_time = 0.0

        self.publisher = self.create_publisher(
            JointState,
            output_topic,
            qos_profile_sensor_data,
        )
        self.object_publisher = self.create_publisher(
            String,
            f"/{namespace}/mujoco/object_states",
            10,
        )
        self.pingpong_state_publisher = self.create_publisher(
            String, f"/{namespace}/pingpong/state", 10
        )
        self._pingpong_state_period = 1.0 / max(
            float(self.get_parameter("pingpong_state_rate").value), 1.0
        )
        self._last_pingpong_state = 0.0
        self.motor_diag_publisher = self.create_publisher(
            String,
            f"/{namespace}/motor/diagnostics",
            10,
        )
        self._last_arm_tau = np.zeros(len(_ARM_JOINTS))
        self._last_arm_speed = np.zeros(len(_ARM_JOINTS))
        self.subscription = self.create_subscription(
            JointState,
            input_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        self.joint_states_publisher = None
        if self.authority_mode:
            js_topic = str(self.get_parameter("joint_states_topic").value).strip()
            js_topic = js_topic or f"/{namespace}/joint_states"
            self.joint_states_publisher = self.create_publisher(
                JointState, js_topic, qos_profile_sensor_data
            )
            self.get_logger().info(f"authority mode: publishing joint states on {js_topic}")

        self.command_subscriptions = []
        if self.subscribe_commands:
            command_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            if bool(self.get_parameter("legacy_joint_command_topics").value):
                from rebotarm_msgs.msg import JointPosVelCmd
                for index, name in enumerate(_ARM_JOINTS):
                    self.command_subscriptions.append(self.create_subscription(
                        JointPosVelCmd,
                        f"/{namespace}/joints/{name}/cmd/pos_vel",
                        self._command_callback(index), command_qos))
                self.get_logger().info("legacy per-joint command topics enabled")
            else:
                self.command_subscriptions.append(self.create_subscription(
                    ArmPositionCommand,
                    f"/{namespace}/arm/command/position",
                    self._arm_command_callback,
                    command_qos,
                ))
                self.get_logger().info(
                    f"subscribing to /{namespace}/arm/command/position for targets"
                )

        if self.control_mode == "cartesian":
            target_topic = (
                str(self.get_parameter("cartesian_target_topic").value).strip()
                or f"/{namespace}/cartesian_target"
            )
            self.create_subscription(
                PoseStamped, target_topic, self._cartesian_target_callback, 10
            )
            self.get_logger().info(f"cartesian impedance target: {target_topic}")

        if self.provide_gripper_services:
            self.create_service(
                SetGripper, f"/{namespace}/gripper/set", self._srv_set_gripper
            )
            self.create_service(
                GripperCommand, f"/{namespace}/gripper/open", self._srv_open_gripper
            )
            self.create_service(
                GripperCommand, f"/{namespace}/gripper/close", self._srv_close_gripper
            )
            self.get_logger().info("providing gripper services")

        self.reset_service = self.create_service(
            Trigger,
            f"/{namespace}/mujoco/reset",
            self._reset,
        )

        self.viewer = None
        self._last_viewer_sync = 0.0
        self._viewer_period = 1.0 / max(float(self.get_parameter("viewer_rate").value), 1.0)
        if bool(self.get_parameter("use_viewer").value):
            from mujoco import viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

        self.timer = self.create_timer(1.0 / self.update_rate, self._update)
        self.get_logger().info(
            f"RS MuJoCo ready: mode={self.simulation_mode}, input={input_topic}, "
            f"output={output_topic}, model={self.model_path}"
        )

    def _vector_parameter(self, name: str) -> np.ndarray:
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (6,):
            raise ValueError(f"{name} must contain 6 values")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values")
        return values

    def _clamp_arm_target(self, target: np.ndarray) -> np.ndarray:
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        return np.clip(np.asarray(target, dtype=np.float64), lower, upper)

    def _required_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"required MuJoCo object not found: {name}")
        return int(object_id)

    @staticmethod
    def _default_motor_specs_path() -> str:
        return str(description_path() / "config/motor_specs.yaml")

    def _apply_motor_model_to_model(self) -> None:
        assert self.motor_model is not None
        for index, name in enumerate(_ARM_JOINTS):
            if name not in self.motor_model.joint_names:
                continue
            spec = self.motor_model.spec(name)
            dof = int(self.arm_dof_addrs[index])
            self.model.dof_armature[dof] += self.motor_model.reflected_inertia(name)
            self.model.dof_frictionloss[dof] += spec.friction_static
            self.model.dof_damping[dof] += spec.friction_viscous

    def _joint_state_callback(self, msg: JointState) -> None:
        values = dict(zip(msg.name, msg.position))
        for index, name in enumerate(_ARM_JOINTS):
            if name in values and np.isfinite(values[name]):
                self.target_arm[index] = float(values[name])
        self.target_arm = self._clamp_arm_target(self.target_arm)

        if "gripper_joint1" in values:
            self.target_gripper = self._visual_to_mujoco_gripper(
                values["gripper_joint1"], _RS_ROS_VISUAL_OPEN_M
            )
        elif "finger_left" in values:
            self.target_gripper = self._visual_to_mujoco_gripper(
                values["finger_left"], _LEGACY_VISUAL_OPEN_M
            )
        self.last_input_time = time.monotonic()

    def _command_callback(self, index: int):
        def callback(msg) -> None:
            if np.isfinite(msg.pos) and np.isfinite(msg.vlim) and msg.vlim >= 0.0:
                target = self.target_arm.copy()
                target[index] = float(msg.pos)
                self.target_arm = self._clamp_arm_target(target)
                self._arm_command_speed[index] = min(float(msg.vlim), self.arm_target_speed[index])
                self.last_input_time = time.monotonic()

        return callback

    def _arm_command_callback(self, msg: ArmPositionCommand) -> None:
        try:
            positions, speeds = validate_position_command(msg.positions, msg.velocity_limits)
        except ValueError:
            return
        self.target_arm = self._clamp_arm_target(positions)
        # 与 STM32 一致：0 暂停该轴位置轨迹，不抬高到最小正速度。
        self._arm_command_speed = np.minimum(speeds, self.arm_target_speed)
        self.last_input_time = time.monotonic()

    def _cartesian_target_callback(self, msg: PoseStamped) -> None:
        self.cartesian_target = msg
        self.last_input_time = time.monotonic()

    # ------------------------------------------------------------------
    # gripper services (authority mode)
    # ------------------------------------------------------------------

    def _set_gripper_target_rad(self, position_rad: float) -> None:
        ratio = float(np.clip((position_rad - _GRIPPER_MOTOR_CLOSE_RAD) /
                              (_GRIPPER_MOTOR_OPEN_RAD - _GRIPPER_MOTOR_CLOSE_RAD), 0.0, 1.0))
        self.target_gripper = float(
            np.clip(ratio * _MUJOCO_GRIPPER_OPEN_M, 0.0, _MUJOCO_GRIPPER_OPEN_M)
        )

    def _current_gripper_rad(self) -> float:
        # joint7 master travel in metres maps linearly to motor radians.
        opening = float(self.data.qpos[self.master_qpos_addr])
        frac = float(np.clip(opening / _MUJOCO_GRIPPER_OPEN_M, 0.0, 1.0))
        return _GRIPPER_MOTOR_CLOSE_RAD + frac * (_GRIPPER_MOTOR_OPEN_RAD - _GRIPPER_MOTOR_CLOSE_RAD)

    def _srv_set_gripper(self, request, response):
        self._set_gripper_target_rad(float(request.position))
        requested_limit = max(float(request.max_effort), 0.0)
        self._command_gripper_force_limit = (
            requested_limit if requested_limit > 0.0 else self.gripper_force_limit
        )
        response.success = True
        response.reached_position = self._current_gripper_rad()
        return response

    def _srv_open_gripper(self, request, response):
        position = float(request.position)
        if position <= 0.0:
            position = _GRIPPER_MOTOR_OPEN_RAD
        self._set_gripper_target_rad(position)
        response.success = True
        response.reached_position = self._current_gripper_rad()
        response.message = "gripper open accepted"
        return response

    def _srv_close_gripper(self, request, response):
        self._set_gripper_target_rad(float(request.position))
        response.success = True
        response.reached_position = self._current_gripper_rad()
        response.message = "gripper close accepted"
        return response

    # ------------------------------------------------------------------
    # cartesian impedance
    # ------------------------------------------------------------------

    def _cartesian_torque(self) -> np.ndarray:
        assert self.cartesian_target is not None
        d = self.data
        tcp_id = self._tcp_body_id
        point = d.xpos[tcp_id].copy()
        rotation = d.xmat[tcp_id].reshape(3, 3).copy()
        mujoco.mj_jac(self.model, d, self._cart_jacp, self._cart_jacr, point, tcp_id)

        pose = self.cartesian_target.pose
        desired_p = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )
        desired_q = np.array(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(desired_q))
        if norm < 1e-8:
            desired_q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            desired_q /= norm
        desired_R = Rotation.from_quat(desired_q).as_matrix()

        jac = np.vstack(
            [
                self._cart_jacp[:, self.arm_dof_addrs],
                self._cart_jacr[:, self.arm_dof_addrs],
            ]
        )
        twist = jac @ d.qvel[self.arm_dof_addrs]
        position_error = desired_p - point
        orientation_error = Rotation.from_matrix(desired_R @ rotation.T).as_rotvec()

        force = np.concatenate(
            [
                self.cartesian_kp * position_error - self.cartesian_kd * twist[:3],
                self.cartesian_rot_kp * orientation_error
                - self.cartesian_rot_kd * twist[3:],
            ]
        )
        return self.gravity_bias_scale * self.data.qfrc_bias[self.arm_dof_addrs] + jac.T @ force

    @staticmethod
    def _visual_to_mujoco_gripper(position: float, visual_open: float) -> float:
        ratio = np.clip(float(position) / visual_open, 0.0, 1.0)
        return float(ratio * _MUJOCO_GRIPPER_OPEN_M)

    def _update(self) -> None:
        if self.last_input_time == 0.0:
            return

        stale = (
            not self.authority_mode
            and self.stale_timeout > 0.0
            and time.monotonic() - self.last_input_time > self.stale_timeout
        )
        if stale:
            # A real controller should hold its last measured position when its
            # command stream disappears. Continue stepping and publishing state
            # so consumers can observe the watchdog condition instead of seeing
            # a frozen simulation.
            self.target_arm = self.data.qpos[self.arm_qpos_addrs].copy()
            self.target_gripper = float(self.data.qpos[self.master_qpos_addr])
            self._applied_arm_target = self.target_arm.copy()
            self._applied_gripper_target = self.target_gripper
            self.cartesian_target = None

        viewer_lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with viewer_lock:
            if self.simulation_mode == "kinematic":
                self._update_kinematic()
            else:
                self._update_physics()
            self._publish_state()
            self._publish_object_states()
            self._publish_motor_diagnostics(stale=stale)
        if self.viewer is not None:
            if self.viewer.is_running():
                now = time.monotonic()
                if now-self._last_viewer_sync >= self._viewer_period:
                    self.viewer.sync()
                    self._last_viewer_sync = now
            else:
                self.viewer.close()
                self.viewer = None

    def _update_kinematic(self) -> None:
        current = self.data.qpos[self.arm_qpos_addrs]
        next_arm = current + self.smoothing_alpha * (self.target_arm - current)
        self.data.qpos[self.arm_qpos_addrs] = next_arm
        self.data.qvel[self.arm_dof_addrs] = 0.0

        # Drive the master joint and mirror the fingers explicitly (the
        # equality constraint is not applied in kinematic mode).
        current = float(self.data.qpos[self.master_qpos_addr])
        master = current + self.smoothing_alpha * (
            self.target_gripper - current
        )
        self.data.qpos[self.master_qpos_addr] = master
        self.data.qpos[self.left_qpos_addr] = -master
        self.data.qpos[self.right_qpos_addr] = master
        self.data.qvel[
            [self.master_dof_addr, self.left_dof_addr, self.right_dof_addr]
        ] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_gripper_force = 0.0

    @staticmethod
    def _slew_towards(
        current: np.ndarray | float,
        target: np.ndarray | float,
        speed: np.ndarray | float,
        dt: float,
    ) -> np.ndarray | float:
        """Move a target by at most speed * dt, preserving command continuity."""
        current_array = np.asarray(current, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        speed_array = np.maximum(np.asarray(speed, dtype=np.float64), 0.0)
        delta = target_array - current_array
        limited = current_array + np.clip(delta, -speed_array * dt, speed_array * dt)
        if current_array.ndim == 0:
            return float(limited)
        return limited

    def _update_physics(self) -> None:
        timestep = float(self.model.opt.timestep)
        steps = max(1, int(round((1.0 / self.update_rate) / timestep)))
        gain = self.compliance_scale if self.compliance_enabled else 1.0
        kp = self.arm_kp * gain
        kd = self.arm_kd * gain
        for _ in range(steps):
            dt = timestep
            if self.pingpong_enabled:
                self._sync_pingpong_paddle()
            self._applied_arm_target = self._slew_towards(
                self._applied_arm_target,
                self.target_arm,
                self._arm_command_speed,
                dt,
            )
            self._applied_gripper_target = self._slew_towards(
                self._applied_gripper_target,
                self.target_gripper,
                self.gripper_target_speed,
                dt,
            )
            q = self.data.qpos[self.arm_qpos_addrs]
            qd = self.data.qvel[self.arm_dof_addrs]
            if self.control_mode == "cartesian" and self.cartesian_target is not None:
                tau = self._cartesian_torque()
            else:
                gravity_bias = self.data.qfrc_bias[self.arm_dof_addrs]
                tau = pd_bias_torque(q, qd, self._applied_arm_target, kp, kd,
                                     gravity_bias, self.gravity_bias_scale)
            tau = limit_arm_torque(tau, qd, self.arm_tau_limit, self.motor_model)
            if self.motor_model is not None:
                for index, name in enumerate(_ARM_JOINTS):
                    self.motor_model.thermal_step(name, float(tau[index]), timestep)
            self._last_arm_tau = tau.copy()
            self._last_arm_speed = qd.copy()
            self.data.ctrl[self.arm_actuator_ids] = tau

            # Drive the single master joint; the equality syncs the fingers.
            master_q = float(self.data.qpos[self.master_qpos_addr])
            master_qd = float(self.data.qvel[self.master_dof_addr])
            master_tau = self.gripper_kp * (
                self._applied_gripper_target - master_q
            ) - self.gripper_kd * master_qd
            master_tau = float(
                np.clip(master_tau, -self.gripper_tau_limit, self.gripper_tau_limit)
            )
            if self._command_gripper_force_limit > 0.0:
                master_tau = float(
                    np.clip(
                        master_tau,
                        -self._command_gripper_force_limit,
                        self._command_gripper_force_limit,
                    )
                )
            self.data.ctrl[self.master_actuator_id] = master_tau
            mujoco.mj_step(self.model, self.data)
            if self.pingpong_enabled:
                self._publish_pingpong_state()

        # Constraint (contact) forces are populated by the last mj_step.
        self._last_arm_tau_ext = self.data.qfrc_constraint[self.arm_dof_addrs].copy()
        self._last_gripper_force = float(
            self.data.actuator_force[self.master_actuator_id]
        )

    def _sync_pingpong_paddle(self) -> None:
        """Make the simulated paddle rigidly follow the real end-effector pose."""
        end_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
        mocap_id = int(self.model.body_mocapid[self.pingpong_paddle_body])
        if mocap_id < 0:
            return
        end_rotation = self.data.xmat[end_id].reshape(3, 3)
        # Leave clearance from the fingers; the paddle is mounted on the
        # gripper tool flange, not through the finger collision geometry.
        self.data.mocap_pos[mocap_id] = self.data.xpos[end_id] + end_rotation[:, 2] * 0.08
        self.data.mocap_quat[mocap_id] = Rotation.from_matrix(end_rotation).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)

    def _publish_pingpong_state(self) -> None:
        now = time.monotonic()
        if now - self._last_pingpong_state < self._pingpong_state_period:
            return
        ball = self.pingpong_ball_body
        paddle = self.pingpong_paddle_body
        payload = {
            "stamp": now,
            "ball_position_m": self.data.xpos[ball].tolist(),
            "ball_velocity_mps": self.data.cvel[ball][3:].tolist(),
            "paddle_position_m": self.data.xpos[paddle].tolist(),
            "paddle_normal": self.data.xmat[paddle].reshape(3, 3)[:, 2].tolist(),
            "contacts": int(self.data.ncon),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.pingpong_state_publisher.publish(msg)
        self._last_pingpong_state = now

    def _publish_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [*_ARM_JOINTS, "gripper_joint1", "gripper_joint2"]
        msg.position = [
            *[float(value) for value in self.data.qpos[self.arm_qpos_addrs]],
            float(self.data.qpos[self.left_qpos_addr]),
            float(self.data.qpos[self.right_qpos_addr]),
        ]
        msg.velocity = [
            *[float(value) for value in self.data.qvel[self.arm_dof_addrs]],
            float(self.data.qvel[self.left_dof_addr]),
            float(self.data.qvel[self.right_dof_addr]),
        ]
        msg.effort = [
            *[float(value) for value in self.data.qfrc_actuator[self.arm_dof_addrs]],
            float(self._last_gripper_force),
            float(self._last_gripper_force),
        ]
        self.publisher.publish(msg)
        if self.joint_states_publisher is not None:
            self.joint_states_publisher.publish(msg)
        self.last_publish_time = time.monotonic()

    def _publish_object_states(self) -> None:
        now = time.monotonic()
        if now - self.last_object_publish_time < self.object_publish_period:
            return
        contacts_by_object = {
            body_id: {"fingers": set(), "support": set(), "pairs": []}
            for body_id in self.object_body_ids.values()
        }
        for contact in self.data.contact:
            if float(contact.dist) > 0.001:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            for object_body, other_body, object_geom, other_geom in (
                (body1, body2, geom1, geom2),
                (body2, body1, geom2, geom1),
            ):
                state = contacts_by_object.get(object_body)
                if state is None:
                    continue
                other_name = self.model.body(other_body).name or "world"
                finger_name = self._finger_names_by_id.get(other_body)
                if finger_name is not None:
                    state["fingers"].add(finger_name)
                else:
                    state["support"].add(other_name)
                state["pairs"].append(
                    {
                        "object_geom": self.model.geom(object_geom).name,
                        "other_geom": self.model.geom(other_geom).name,
                        "other_body": other_name,
                        "distance_m": round(float(contact.dist), 6),
                    }
                )

        objects = []
        for name, body_id in self.object_body_ids.items():
            contact_state = contacts_by_object[body_id]
            objects.append(
                {
                    "name": name,
                    "position": [float(value) for value in self.data.xpos[body_id]],
                    "quaternion": [float(value) for value in self.data.xquat[body_id]],
                    "finger_contacts": sorted(contact_state["fingers"]),
                    "support_contacts": sorted(contact_state["support"]),
                    "contact_pairs": contact_state["pairs"],
                }
            )
        msg = String()
        msg.data = json.dumps(
            {"objects": objects, "simulation_mode": self.simulation_mode},
            separators=(",", ":"),
        )
        self.object_publisher.publish(msg)
        self.last_object_publish_time = now

    def _publish_motor_diagnostics(self, *, stale: bool = False) -> None:
        if self.motor_model is None:
            return
        joints: dict[str, dict] = {}
        for index, name in enumerate(_ARM_JOINTS):
            if name not in self.motor_model.joint_names:
                continue
            spec = self.motor_model.spec(name)
            tau = float(self._last_arm_tau[index])
            speed = float(self._last_arm_speed[index])
            limit = self.motor_model.torque_limit(name, speed)
            temperature = self.motor_model.temperature(name)
            tau_ext = float(self._last_arm_tau_ext[index])
            joints[name] = {
                "motor": spec.motor,
                "motor_mass": spec.mass,
                "tau": round(tau, 4),
                "tau_rated": spec.rated,
                "tau_peak": spec.peak,
                "tau_limit": round(limit, 4),
                "speed": round(speed, 4),
                "temperature": round(temperature, 2),
                "saturated": bool(abs(tau) >= limit - 1e-6),
                "utilization_rated": round(abs(tau) / max(spec.rated, 1e-6), 3),
                "external_torque": round(tau_ext, 4),
                "contact": bool(abs(tau_ext) >= self.contact_torque_threshold),
            }
        msg = String()
        msg.data = json.dumps(
            {
                "simulation_mode": self.simulation_mode,
                "input_stale": bool(stale),
                "compliance": {
                    "enabled": self.compliance_enabled,
                    "scale": self.compliance_scale,
                },
                "gripper_force": round(self._last_gripper_force, 4),
                "gripper_force_limit": round(self._command_gripper_force_limit, 4),
                "joints": joints,
            },
            separators=(",", ":"),
        )
        self.motor_diag_publisher.publish(msg)

    def _reset(self, _request, response):
        mujoco.mj_resetData(self.model, self.data)
        self.target_arm.fill(0.0)
        self.target_gripper = 0.0
        self._command_gripper_force_limit = self.gripper_force_limit
        self._applied_arm_target = self.target_arm.copy()
        self._applied_gripper_target = 0.0
        self._arm_command_speed = self.arm_target_speed.copy()
        self.last_input_time = time.monotonic() if self.authority_mode else 0.0
        mujoco.mj_forward(self.model, self.data)
        response.success = True
        response.message = "RS MuJoCo reset"
        return response

    def destroy_node(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RsMujocoSync()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
