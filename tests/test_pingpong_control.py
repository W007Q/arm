"""Controller isolation and causal actuator limits for physical transfer."""
import copy

import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from rebotarm_description.parameters import description_path, uart_parameters
from rebotarm_pingpong.control import CartesianServo, horizontal_home


@pytest.fixture
def servo_and_plant():
    root = description_path()
    config = yaml.safe_load((root / "config/pingpong.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(root / "mjcf/pingpong_3d_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:6] = horizontal_home(model, config)
    mujoco.mj_forward(model, data)
    return CartesianServo(model, config), model, data, config


def test_nominal_controller_is_independent_of_ball_contacts_and_plant_inertia(servo_and_plant):
    servo, model, data, _ = servo_and_plant
    position, velocity, rotation = servo.paddle_state(data)
    np.testing.assert_allclose(position, data.site_xpos[servo.site], atol=1e-12)
    np.testing.assert_allclose(velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rotation, data.site_xmat[servo.site].reshape(3, 3), atol=1e-12)

    def torque_trace():
        values = []
        for now in (0., .002, .004, .006, .008):
            data.time = now
            servo.apply(data, .20)
            values.append(data.ctrl.copy())
        return np.array(values)

    reference = torque_trace()
    servo.reset()
    ball = model.joint("pingpong_ball_freejoint")
    ball_q, ball_v = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[ball_q:ball_q+3] = [2., -3., .1]
    data.qvel[ball_v:ball_v+6] = 30.
    # Live plant oracles must not enter nominal controller dynamics.
    data.qfrc_bias[:] = 1e8
    model.body_mass[1:7] *= 1.25
    np.testing.assert_allclose(servo.paddle_state(data)[0], position, atol=1e-12)
    np.testing.assert_allclose(torque_trace(), reference, rtol=1e-12, atol=1e-12)
    assert servo.nominal_data.ncon == 0


def test_actuator_delay_and_lag_are_causal_and_reset_clears_them(servo_and_plant):
    servo, model, data, config = servo_and_plant
    for now in (0., .002):
        data.time = now
        servo.apply(data, .20)
        np.testing.assert_array_equal(data.ctrl, np.zeros(6))
    data.time = .004
    servo.apply(data, .20)
    delayed_response = data.ctrl.copy()
    assert np.max(np.abs(delayed_response)) > 0
    assert np.all(np.abs(delayed_response) <= servo.peak_torque)

    fast_config = copy.deepcopy(config)
    fast_config["control"]["actuator_time_constant_s"] = 1e-6
    fast = CartesianServo(model, fast_config)
    for now in (0., .002, .004):
        data.time = now
        fast.apply(data, .20)
    assert np.linalg.norm(delayed_response) < np.linalg.norm(data.ctrl)

    servo.reset()
    assert not servo.pending_commands
    np.testing.assert_array_equal(servo.applied_torque, np.zeros(6))
    data.time = 0.
    servo.apply(data, .20)
    np.testing.assert_array_equal(data.ctrl, np.zeros(6))


def test_speed_envelope_brakes_overspeed_without_changing_velocity(servo_and_plant):
    servo, _, data, _ = servo_and_plant
    overspeed = 1.1 * servo.no_load_speed
    np.testing.assert_array_equal(servo._motor_output(servo.peak_torque, overspeed), np.zeros(6))
    np.testing.assert_array_equal(servo._motor_output(-servo.peak_torque, -overspeed), np.zeros(6))
    np.testing.assert_allclose(servo._motor_output(-servo.peak_torque, overspeed), -servo.peak_torque)
    np.testing.assert_allclose(servo._motor_output(servo.peak_torque, -overspeed), servo.peak_torque)

    data.qvel[:6] = .70
    actual_velocity = data.qvel.copy()
    for index in range(20):
        data.time = index / 500.
        servo.apply(data, .60)
    # Limit targets; never overwrite the physically integrated velocity.
    np.testing.assert_array_equal(data.qvel, actual_velocity)
    target_velocity = servo.path(servo.reference[2], 1) * servo.velocity
    assert np.all(np.abs(target_velocity) <= servo.joint_speed_limit * 1.01)


def test_host_transport_cadence_holds_targets_between_packets(servo_and_plant):
    servo, _, data, _ = servo_and_plant
    assert servo.command_hz == uart_parameters()["tx_hz"]
    data.time = 0.
    servo.apply(data, .20)
    assert servo.transmitted_commands == 1
    first_target = servo.pending_commands[0][1].copy()

    data.time = .002
    servo.apply(data, .20)
    assert servo.transmitted_commands == 1
    assert len(servo.pending_commands) == 1
    np.testing.assert_array_equal(servo.pending_commands[0][1], first_target)

    data.time = servo.command_period_s
    servo.apply(data, .20)
    assert servo.transmitted_commands == 2
    # The first packet arrived; the second packet is still in transit.
    np.testing.assert_array_equal(servo.delayed_position, first_target)
    assert len(servo.pending_commands) == 1
    assert np.linalg.norm(servo.pending_commands[0][1] - first_target) > 0

    # Missing host ticks never invent packets for historical simulation times.
    data.time = 5. * servo.command_period_s
    servo.apply(data, .20)
    assert servo.transmitted_commands == 3
    assert servo.next_tx_time > data.time
    servo.reset()
    assert servo.transmitted_commands == 0 and servo.next_tx_time == 0.


def test_controller_respects_each_firmware_joint_speed_limit(servo_and_plant):
    servo, _, _, _ = servo_and_plant
    transport_limits = np.asarray(uart_parameters()["velocity_limit"][:6])
    assert np.all(servo.joint_speed_limit <= transport_limits)
    # Firmware slots are reordered to logical joint order before use.
    np.testing.assert_allclose(transport_limits, [.3, .3, .3, .3, .5, .3])


def test_cartesian_target_uses_camera_axes_and_rejects_invalid_track(servo_and_plant):
    from types import SimpleNamespace
    servo, _, _, _ = servo_and_plant
    # Velocity along the local paddle X direction must ask for a negative
    # local pitch.  Its horizontal normal points against the incoming drift.
    world_velocity = servo.home_rotation @ np.array([.20, 0., 0.])
    estimate = SimpleNamespace(valid=True, position=servo.home_xyz + [0., 0., .2],
                               velocity=world_velocity)
    target, tilt = servo.target_from_ball_estimate(estimate)
    assert abs(tilt[0]) < 1e-12 and tilt[1] < 0
    normal = servo.home_rotation @ Rotation.from_euler("xy", tilt).as_matrix()[:, 2]
    assert normal[:2] @ world_velocity[:2] < 0
    assert servo.target_from_ball_estimate(SimpleNamespace(valid=False)) is None
    assert np.linalg.norm(target[:2]-servo.home_xyz[:2]) <= .02


def test_cartesian_trajectory_respects_workspace_and_joint_target_speed(servo_and_plant):
    servo, model, data, config = servo_and_plant
    old_q = servo._last_ik_command_q.copy()
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    for index in range(200):
        data.time = index / config["control"]["servo_hz"]
        servo.apply_cartesian(data, [5., -5., 5.], [5., -5.])
        delta = abs(servo._last_ik_command_q-old_q)
        assert np.all(delta <= servo.joint_speed_limit/config["control"]["servo_hz"] + 1e-12)
        old_q = servo._last_ik_command_q.copy()
        assert np.all(abs(servo.reference[:2]-servo.home_xyz[:2]) <= servo.xy_tracking_limit+1e-12)
        assert abs(servo.reference[2]-servo.home_xyz[2]) <= config["control"]["stroke_m"]+1e-12
        assert np.all(abs(servo.cartesian_tilt) <= servo.tilt_limit_rad+1e-12)
        assert np.all(abs(data.ctrl) <= servo.peak_torque+1e-12)
    # This controller only writes actuator torque.  It never changes the
    # state as a way of making the commanded pose or velocity happen.
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
