"""SDK and fake adapters: complete arm updates without opening a motor bus."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rebotarm_description.parameters import ARM_JOINTS, joint_limits
from rebotarmcontroller.conversions import fk_to_pose, pose_to_xyz_rpy
from rebotarmcontroller.fake_rs_driver import FakeRsDriver
from rebotarmcontroller.hardware_manager import HardwareManager
from rebotarmcontroller.motor_passthrough import MotorPassthrough


def sdk_hardware(mode="mit"):
    hardware = object.__new__(HardwareManager)
    hardware._cmd_lock = threading.RLock()
    hardware._enabled = True
    hardware._gravity_comp_active = False
    hardware._state_machine = "IDLE"
    hardware._batch_position_active = False
    hardware._arm_control_mode = mode
    hardware._control_output_enabled = True
    hardware._arm_mit_kp = np.array([3.1, 3.1, 3.1, 1.2, 2.1, 1.2])
    hardware._arm_mit_kd = np.array([1.4, 1.4, 1.4, .6, .3, .6])
    limits = joint_limits()
    model = SimpleNamespace(
        lowerPositionLimit=np.array([limits[name]["lower"] for name in ARM_JOINTS]),
        upperPositionLimit=np.array([limits[name]["upper"] for name in ARM_JOINTS]),
    )
    hardware._endpos_ctrl = SimpleNamespace(_model=model, _running=False)
    sent, starts, stops, gripper_ticks = [], [], [], []
    hardware._mode_changes = []
    group = SimpleNamespace(
        joint_names=list(ARM_JOINTS), mode="mit",
        get_positions=lambda **_: np.zeros(6),
        send_pos_vel=lambda positions, vlim: sent.append(
            dict(mode="pos_vel", positions=np.array(positions, copy=True),
                 vlim=np.array(vlim, copy=True))
        ),
        send_mit=lambda positions, **kwargs: sent.append(
            dict(mode="mit", positions=np.array(positions, copy=True),
                 **{key: np.array(value, copy=True) for key, value in kwargs.items()})
        ),
    )

    def pos_vel_mode():
        hardware._mode_changes.append("pos_vel")
        group.mode = "pos_vel"
        return True

    def mit_mode(**_):
        hardware._mode_changes.append("mit")
        group.mode = "mit"
        return True

    group.mode_pos_vel = pos_vel_mode
    group.mode_mit = mit_mode
    hardware._arm_group = group
    hardware._robot = SimpleNamespace(control_loop_active=False)

    def start(callback):
        starts.append(callback)
        hardware._robot.control_loop_active = True

    def stop():
        stops.append(True)
        hardware._robot.control_loop_active = False

    hardware._robot.start_control_loop = start
    hardware._robot.stop_control_loop = stop
    hardware._send_gripper_bounded = lambda: gripper_ticks.append(True)
    return hardware, sent, starts, stops, gripper_ticks


def test_sdk_posvel_batch_updates_all_axes_once_and_reuses_control_loop():
    hardware, sent, starts, stops, _ = sdk_hardware("posvel")
    positions = [.1, -.2, .3, -.1, .2, -.3]
    vlims = [.1, .2, .3, .2, .1, 0.]
    hardware.send_arm_position_command(positions, vlims)
    assert sent[0]["mode"] == "pos_vel"
    np.testing.assert_allclose(sent[0]["positions"], positions)
    np.testing.assert_allclose(sent[0]["vlim"], vlims)
    positions[0] = .2
    hardware.send_arm_position_command(positions, vlims)
    assert len(starts) == len(stops) == 1
    assert hardware.state_machine == "LOWLEVEL_STREAMING"
    np.testing.assert_allclose(sent[1]["positions"], positions)


def test_sdk_default_mit_keeps_gains_and_slews_reference_at_requested_limits():
    hardware, sent, starts, stops, _ = sdk_hardware()
    positions = [.1, -.2, .3, -.1, .2, -.3]
    speeds = [.1, .2, .3, .2, .1, 0.]
    hardware.send_arm_position_command(positions, speeds)
    # No instantaneous jump on receipt: the reference starts at measured zero.
    np.testing.assert_allclose(sent[0]["positions"], 0.)
    hardware._batch_position_loop_cb(hardware._robot, .1)
    np.testing.assert_allclose(sent[-1]["positions"], [.01, -.02, .03, -.02, .01, 0.])
    assert sent[-1]["mode"] == hardware.mode == "mit"
    np.testing.assert_allclose(sent[-1]["vel"], 0.)
    np.testing.assert_allclose(sent[-1]["kp"], hardware._arm_mit_kp)
    np.testing.assert_allclose(sent[-1]["kd"], hardware._arm_mit_kd)
    assert hardware._mode_changes == []

    # A later batch updates atomically without resetting the limited reference
    # or restarting control. A zero limit pauses even when the goal changes.
    hardware.send_arm_position_command([.5, -.2, .3, -.1, .2, .4], [0., .2, .3, .2, .1, 0.])
    hardware._batch_position_loop_cb(hardware._robot, .1)
    np.testing.assert_allclose(sent[-1]["positions"], [.01, -.04, .06, -.04, .02, 0.])
    assert len(starts) == len(stops) == 1


def test_sdk_mit_applies_configured_speed_ceilings():
    hardware, sent, _, _, _ = sdk_hardware()
    hardware._arm_group._pv_vlim = np.full(6, .05)
    hardware.send_arm_position_command([.5, -.5, .5, .5, .5, .5], [.2]*6)
    hardware._batch_position_loop_cb(hardware._robot, .1)
    np.testing.assert_allclose(sent[-1]["positions"], [.005, -.005, .005, .005, .005, .005])


@pytest.mark.parametrize("positions, vlims", [
    ([0.]*5, [.2]*6),
    ([float("nan")]*6, [.2]*6),
    ([0.]*6, [-.1]*6),
    ([0.]*6, [float("inf")]*6),
    ([0., 1., 0., 0., 0., 0.], [.2]*6),  # joint2 is non-positive in URDF
])
def test_sdk_rejects_invalid_batch_before_switching_modes(positions, vlims):
    hardware, sent, starts, stops, _ = sdk_hardware()
    with pytest.raises(ValueError):
        hardware.send_arm_position_command(positions, vlims)
    assert not sent and not starts and not stops
    assert hardware.mode == "mit"


def test_sdk_rejects_wrong_joint_order_and_disabled_arm():
    hardware, sent, starts, _, _ = sdk_hardware()
    hardware._arm_group.joint_names.reverse()
    with pytest.raises(ValueError, match="joint1..joint6"):
        hardware.send_arm_position_command([0.]*6, [.2]*6)
    hardware._arm_group.joint_names.reverse()
    hardware._enabled = False
    with pytest.raises(RuntimeError, match="disabled"):
        hardware.send_arm_position_command([0.]*6, [.2]*6)
    assert not sent and not starts


@pytest.mark.parametrize("mode", ["mit", "posvel"])
def test_sdk_expired_stream_holds_measured_position_and_keeps_gripper_loop(mode):
    hardware, sent, _, _, gripper_ticks = sdk_hardware(mode)
    hardware.send_arm_position_command([.1, -.1, .1, .1, .1, .1], [.2]*6)
    hardware._stream_at = time.monotonic() - 1.
    hardware._batch_position_loop_cb(hardware._robot, .01)
    np.testing.assert_allclose(sent[-1]["positions"], np.zeros(6))
    assert hardware.state_machine == "IDLE"
    assert hardware.enabled and hardware._robot.control_loop_active
    assert gripper_ticks == [True]


def passthrough(source):
    subscriptions, commands, warnings = {}, [], []
    hardware = SimpleNamespace(
        joint_names=list(ARM_JOINTS), has_gripper=True, state_machine="IDLE",
        send_arm_position_command=lambda *args: commands.append(args),
    )
    node = SimpleNamespace(
        reentrant_group=None,
        create_subscription=lambda _, topic, callback, qos, **kwargs:
            subscriptions.setdefault(topic, callback),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
        publish_arm_status=lambda: None,
    )
    MotorPassthrough(node, hardware, "rebotarm", "reject", arm_command_source=source)
    return subscriptions, commands


def test_sdk_command_sources_are_mutually_exclusive():
    subscriptions, commands = passthrough("batch")
    assert "/rebotarm/arm/command/position" in subscriptions
    assert not any("/joints/" in topic for topic in subscriptions)
    message = SimpleNamespace(positions=[0.]*6, velocity_limits=[.2]*6, stamp=None)
    subscriptions["/rebotarm/arm/command/position"](message)
    assert len(commands) == 1
    np.testing.assert_allclose(commands[0][0], message.positions)
    np.testing.assert_allclose(commands[0][1], message.velocity_limits)
    assert commands[0][2] is None
    subscriptions, _ = passthrough("per_joint")
    assert "/rebotarm/arm/command/position" not in subscriptions
    assert sum("/joints/" in topic for topic in subscriptions) == 12


def fake_driver():
    driver = object.__new__(FakeRsDriver)
    driver.enabled = True
    driver.state_machine = "IDLE"
    driver.max_joint_speed = .5
    driver.positions = [0.]*6
    driver.targets = [0.]*6
    driver.target_speeds = [.5]*6
    limits = joint_limits()
    driver._joint_limits = [(limits[name]["lower"], limits[name]["upper"])
                            for name in ARM_JOINTS]
    driver.publish_status = lambda: None
    driver.get_logger = lambda: SimpleNamespace(warn=lambda _: None)
    return driver


def test_fake_batch_zero_speed_holds_and_reads_real_joint_limits():
    driver = fake_driver()
    driver.positions[0] = .05
    driver._arm_position_callback(SimpleNamespace(
        positions=[1., -10., .2, .3, .4, .5],
        velocity_limits=[0., .2, .3, .4, .5, 1.],
    ))
    assert driver.targets[0] == .05
    assert driver.target_speeds[0] == 0.
    assert driver.targets[1] == driver._joint_limits[1][0]
    assert driver.target_speeds == pytest.approx([0., .2, .3, .4, .5, .5])


def test_fake_invalid_batch_cannot_partially_overwrite_previous_target():
    driver = fake_driver()
    driver.targets = [.1]*6
    driver._arm_position_callback(SimpleNamespace(
        positions=[.2]*6, velocity_limits=[.2]*5 + [float("nan")],
    ))
    assert driver.targets == [.1]*6
    driver._joint_pos_vel_callback(0)(SimpleNamespace(pos=.5, vlim=0.))
    assert driver.targets[0] == driver.positions[0]


def test_fk_pose_round_trip_uses_ros_xyzw_quaternions():
    position = np.array([.1, -.2, .3])
    rotation = Rotation.from_euler("xyz", [.3, -.4, .5]).as_matrix()
    pose = fk_to_pose(position, rotation)
    xyz_rpy = pose_to_xyz_rpy(pose)
    np.testing.assert_allclose(xyz_rpy[:3], position, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", xyz_rpy[3:]).as_matrix(), rotation, atol=1e-12
    )
