from __future__ import annotations

from rclpy.qos import QoSProfile, ReliabilityPolicy
from rebotarm_msgs.msg import (
    ArmPositionCommand,
    JointMitCmd,
    JointPosVelCmd,
)
from rebotarm_description.command_interface import validate_position_command


class MotorPassthrough:
    """Forward ROS command topics to one hardware adapter.

    ``batch`` is the production interface: one message contains the complete
    six-joint arm target, so a callback can validate and apply it atomically.
    ``per_joint`` remains available for old clients, but the two sources are
    deliberately mutually exclusive to avoid one topic overwriting another.
    """

    def __init__(
        self,
        node,
        hardware,
        namespace: str,
        arbitration: str,
        *,
        arm_command_source: str = "batch",
    ) -> None:
        self._node = node
        self._hardware = hardware
        self._arbitration = arbitration
        self._arm_command_source = str(arm_command_source or "batch").strip().lower()
        if self._arm_command_source not in ("batch", "per_joint"):
            self._node.get_logger().warn(
                f"unsupported arm_command_source={arm_command_source!r}; using 'batch'"
            )
            self._arm_command_source = "batch"
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._subscriptions = []

        joint_commands = (
            (
                JointMitCmd,
                "cmd/mit",
                lambda hw, name, msg: hw.send_joint_mit_cmd(
                    name,
                    msg.pos,
                    msg.vel,
                    msg.kp,
                    msg.kd,
                    msg.tau,
                ),
            ),
            (
                JointPosVelCmd,
                "cmd/pos_vel",
                lambda hw, name, msg: hw.send_joint_pos_vel_cmd(
                    name,
                    msg.pos,
                    msg.vlim,
                ),
            ),
        )
        gripper_commands = (
            (
                JointMitCmd,
                "cmd/mit",
                lambda hw, msg: hw.send_gripper_mit_cmd(
                    msg.pos,
                    msg.vel,
                    msg.kp,
                    msg.kd,
                    msg.tau,
                ),
            ),
            (
                JointPosVelCmd,
                "cmd/pos_vel",
                lambda hw, msg: hw.send_gripper_pos_vel_cmd(msg.pos, msg.vlim),
            ),
        )

        if self._arm_command_source == "batch":
            self._subscribe(
                ArmPositionCommand,
                f"/{namespace}/arm/command/position",
                self._make_batch_callback(f"/{namespace}/arm/command/position"),
                qos,
            )
        else:
            for joint_name in hardware.joint_names:
                for msg_type, label, command in joint_commands:
                    self._subscribe(
                        msg_type,
                        f"/{namespace}/joints/{joint_name}/{label}",
                        self._make_joint_callback(
                            joint_name,
                            label,
                            command,
                        ),
                        qos,
                    )
        if hardware.has_gripper:
            for msg_type, label, command in gripper_commands:
                self._subscribe(
                    msg_type,
                    f"/{namespace}/gripper/{label}",
                    self._make_gripper_callback(label, command),
                    qos,
                )

    def _subscribe(self, msg_type, topic: str, callback, qos: QoSProfile) -> None:
        self._subscriptions.append(
            self._node.create_subscription(
                msg_type,
                topic,
                callback,
                qos,
                callback_group=self._node.reentrant_group,
            )
        )

    def _make_joint_callback(self, joint_name: str, label: str, command) -> object:
        def _callback(msg) -> None:
            if not self._can_send_lowlevel(
                f"/joints/{joint_name}/{label}",
                allow_preempt=True,
            ):
                return

            try:
                command(self._hardware, joint_name, msg)
            except Exception as exc:
                self._node.get_logger().warn(
                    f"joint {label} failed for {joint_name}: {exc}"
                )
            finally:
                self._node.publish_arm_status()

        return _callback

    def _make_batch_callback(self, label: str) -> object:
        def _callback(msg: ArmPositionCommand) -> None:
            try:
                positions, velocity_limits = validate_position_command(
                    msg.positions, msg.velocity_limits
                )
            except ValueError as exc:
                self._node.get_logger().warn(f"batch arm command failed: {exc}")
                return
            if not self._can_send_lowlevel(label, allow_preempt=True):
                return

            try:
                # HardwareManager performs the same validation immediately
                # before sending; doing it there keeps all backends safe even
                # when they are called without this ROS node.
                self._hardware.send_arm_position_command(
                    positions,
                    velocity_limits,
                    msg.stamp,
                )
            except Exception as exc:
                self._node.get_logger().warn(f"batch arm command failed: {exc}")
            finally:
                self._node.publish_arm_status()

        return _callback

    def _make_gripper_callback(self, label: str, command) -> object:
        def _callback(msg) -> None:
            if not self._can_send_lowlevel(
                f"/gripper/{label}",
                allow_preempt=False,
            ):
                return

            try:
                command(self._hardware, msg)
            except Exception as exc:
                self._node.get_logger().warn(f"gripper {label} failed: {exc}")
            finally:
                self._node.publish_arm_status()

        return _callback

    def _can_send_lowlevel(self, label: str, *, allow_preempt: bool) -> bool:
        state = self._hardware.state_machine
        if state in ("GRAVITY_COMP", "SAFE_HOMING"):
            self._node.get_logger().warn(f"rejecting {label} in state {state}")
            return False
        if state == "TRAJ_RUNNING":
            if self._arbitration == "reject" or not allow_preempt:
                self._node.get_logger().warn(
                    f"rejecting {label} while trajectory is running"
                )
                return False
            self._node.get_logger().warn(
                f"preempting trajectory for {label}"
            )
            self._hardware.stop_motion()
        return True
