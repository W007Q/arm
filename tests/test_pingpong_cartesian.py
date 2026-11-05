"""Camera-only Cartesian juggling commands keep the free ball physical."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from rebotarm_description.parameters import description_path
from rebotarm_pingpong.control import CartesianServo, horizontal_home
from rebotarm_pingpong.env import PingPongEnv
from rebotarm_pingpong.vision import BallEstimate


@pytest.fixture
def servo_plant():
    root = description_path()
    config = yaml.safe_load((root / "config/pingpong.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(root / "mjcf/pingpong_3d_scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:6] = horizontal_home(model, config)
    mujoco.mj_forward(model, data)
    return CartesianServo(model, config), model, data, config


def _estimate(x=.31, y=.01, velocity=(.02, -.03, -.8)):
    return BallEstimate(np.array([x, y, .55]), np.array(velocity),
                        True, .012, .8, 0., .012, (None, None))


def test_cartesian_schema_and_camera_only_xy_command(monkeypatch):
    env = PingPongEnv()
    try:
        env.reset(seed=8)
        assert env.observation_space.shape == (43,)
        assert env.action_space.shape == (5,)
        fixed = _estimate()
        monkeypatch.setattr(env.observer, "observe", lambda data: fixed)
        monkeypatch.setattr(env, "truth_ball_state", lambda: (_ for _ in ()).throw(
            AssertionError("controller consulted ball truth")))
        before = env._observation()
        command = env.baseline_command.copy()
        env.data.qpos[env.ball_q:env.ball_q + 3] = [.5, .2, .8]
        env.data.qvel[env.ball_dof:env.ball_dof + 6] = [2., -3., 4., 30., 40., 50.]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_array_equal(env._observation(), before)
        np.testing.assert_array_equal(env.baseline_command, command)
        # Change pixels/estimate while retaining exactly the same hidden state.
        moved = _estimate(.34, -.02)
        monkeypatch.setattr(env.observer, "observe", lambda data: moved)
        env._observation()
        assert env.baseline_command[0] > command[0]
        assert env.baseline_command[1] < command[1]
    finally:
        env.close()


def test_local_yaw_tilt_points_against_incoming_lateral_velocity(servo_plant):
    servo, _, _, _ = servo_plant
    estimate = SimpleNamespace(valid=True, position=servo.home_xyz + [0., 0., .2],
                               velocity=np.array([.2, -.1, -1.]))
    _, tilt = servo.target_from_ball_estimate(estimate, tilt_gain=.2)
    normal = servo.home_rotation @ Rotation.from_euler("xy", tilt).as_matrix()[:, 2]
    # A passive impact along this normal adds lateral impulse opposite to
    # the incoming lateral velocity, regardless of the blade's nominal yaw.
    assert np.dot(normal[:2], estimate.velocity[:2]) < 0.
    local_velocity = (servo.home_rotation.T @ np.r_[estimate.velocity[:2], 0.])[:2]
    np.testing.assert_allclose(tilt, .2 * np.array([local_velocity[1], -local_velocity[0]]))


def test_cartesian_mit_targets_obey_every_tick_speed_limit_without_teleport(servo_plant):
    servo, model, data, config = servo_plant
    ball = model.joint("pingpong_ball_freejoint")
    qadr, dadr = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[qadr:qadr + 7] = [.9, -.8, .7, 1., 0., 0., 0.]
    data.qvel[dadr:dadr + 6] = [.3, -.2, .1, 2., -3., 4.]
    before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
    previous = servo._last_ik_command_q.copy()
    dt = 1. / config["control"]["servo_hz"]
    first_reference = None
    for index in range(200):
        data.time = index * dt
        servo.apply_cartesian(data, [10., -10., 10.], [4., -4.])
        target = servo._last_ik_command_q.copy()
        assert np.all(np.abs(target - previous) <= servo.joint_speed_limit * dt + 1e-12)
        assert np.all(np.abs(servo._ik_target_velocity) <= servo.joint_speed_limit + 1e-12)
        assert np.all(np.abs(data.ctrl) <= servo.peak_torque + 1e-12)
        assert np.all(np.abs(servo.reference - servo.home_xyz) <=
                      [servo.xy_tracking_limit + 1e-12, servo.xy_tracking_limit + 1e-12,
                       config["control"]["stroke_m"] + 1e-12])
        if index == 0:
            first_reference = servo.reference.copy()
        previous = target
    # Target limiting is causal; the servo never overwrites physical state.
    np.testing.assert_array_equal(data.qpos, before_qpos)
    np.testing.assert_array_equal(data.qvel, before_qvel)
    # The first command moves in the requested direction.  Later anti-windup
    # may project a target back when the six-axis arm cannot realize the
    # simultaneous XYZ plus tilt request; that projection is the physical
    # workspace limit, not a ball reset.
    assert first_reference[0] > servo.home_xyz[0]
    assert first_reference[1] < servo.home_xyz[1]


def test_top_contact_is_valid_eccentric_contact_and_sidewall_fails_without_bounce(monkeypatch):
    env = PingPongEnv()
    try:
        env.reset(seed=1, options={"height_m": .15, "velocity_m_s": 0.,
                                   "xy_offset_m": [0., 0.], "horizontal_velocity_m_s": [0., 0.],
                                   "spin_rad_s": [0., 0., 0.]})
        paddle = env.data.site_xpos[env.site].copy()
        rotation = env.data.site_xmat[env.site].reshape(3, 3).copy()
        # Put the centre near the rim, still directly above the physical top
        # face. R-r is not a physical restriction for sphere-plane contact.
        env.data.qpos[env.ball_q:env.ball_q + 3] = paddle + rotation @ [.065, 0., .0197]
        mujoco.mj_forward(env.model, env.data)
        assert env._contact()
        assert not env.edge_contact
        # A true side-wall overlap uses the same geom pair, but a different
        # normal and local point. It must not count as a successful bounce.
        env.data.qpos[env.ball_q:env.ball_q + 3] = paddle + rotation @ [.092, 0., .005]
        env.data.qvel[env.ball_dof:env.ball_dof + 3] = 0.
        mujoco.mj_forward(env.model, env.data)
        assert not env._contact()
        assert env.edge_contact
        _, _, done, _, info = env.step(np.zeros(5))
        assert done
        assert info["termination"] == "paddle_edge_missed"
        assert info["bounces"] == 0
    finally:
        env.close()


def test_old_26_feature_checkpoint_is_rejected_even_with_model_override(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from rebotarm_pingpong.evaluation import load_policy
    old = SimpleNamespace(observation_space=SimpleNamespace(shape=(26,)),
                          action_space=SimpleNamespace(shape=(1,)))
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: old)
    with pytest.raises(ValueError, match="schema is incompatible"):
        load_policy(tmp_path / "old_model.zip", allow_mismatch=True)
