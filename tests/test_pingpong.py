"""Independent tool topology, contact physics and actual dynamic juggling."""
import copy
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from rebotarm_description.parameters import description_path, joint_limits, load_hardware
from rebotarm_description.pingpong_model import derived_pingpong_files

from rebotarm_pingpong.env import PingPong1DEnv
from rebotarm_pingpong.diagnostics import drop_report, camera_report
from rebotarm_pingpong.evaluation import load_policy, model_fingerprint


def test_paddle_variant_preserves_six_arm_joints_and_removes_gripper():
    root = description_path()
    original = mujoco.MjModel.from_xml_path(str(root / "mjcf/grasp_scene.xml"))
    for scene, counts in (("pingpong_1d_scene.xml", (13, 12, 6)), ("pingpong_3d_scene.xml", (13, 12, 6))):
        m = mujoco.MjModel.from_xml_path(str(root / "mjcf" / scene))
        assert (m.nq, m.nv, m.nu) == counts
        assert m.neq == 0
        for name in ("joint7", "gripper_joint1", "gripper_joint2"):
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
        for name in ("gripper_end", "d435_link", "gripper_coupler"):
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) == -1
        assert np.all(m.body_mocapid == -1)
        for i in range(1, 7):
            np.testing.assert_allclose(m.joint(f"joint{i}").range, original.joint(f"joint{i}").range)
            np.testing.assert_allclose(m.body(f"link{i}").mass, original.body(f"link{i}").mass)
        assert m.body("pingpong_ball").mass[0] == pytest.approx(.0027)
        assert m.body("pingpong_paddle").mass[0] == pytest.approx(.205)


def test_shared_arm_change_reaches_pingpong_and_bad_physics_is_rejected():
    root = description_path()
    c = yaml.safe_load((root / "config/pingpong.yaml").read_text())
    limits = joint_limits()
    hardware = load_hardware()
    hardware["motor_catalog"]["4310"]["peak"] = 6.
    files = derived_pingpong_files((root/"mjcf/robot.xml").read_text(), c, hardware, limits)
    assert 'ctrlrange="-6 6"' in files["robot_pingpong.xml"]
    invalid = copy.deepcopy(c)
    invalid["ball"]["mass_kg"] = -1
    with pytest.raises(ValueError, match="mass_kg"):
        derived_pingpong_files((root/"mjcf/robot.xml").read_text(), invalid, hardware, limits)


def test_free_ball_moves_sideways_spins_and_can_miss_the_paddle():
    env = PingPong1DEnv()
    try:
        env.reset(seed=1, options={"height_m": .15, "velocity_m_s": 0.,
                                  "xy_offset_m": [0., 0.], "horizontal_velocity_m_s": [.7, 0.],
                                  "spin_rad_s": [3., 5., 7.]})
        q0 = env.data.qpos[env.ball_q:env.ball_q+7].copy()
        for _ in range(60):
            obs, reward, done, truncated, info = env.step([0.])
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert np.all(env.data.ctrl <= env.model.actuator_ctrlrange[:, 1])
            assert np.all(env.data.ctrl >= env.model.actuator_ctrlrange[:, 0])
            if done or truncated:
                break
        assert done and info["termination"] in ("paddle_missed", "ball_side_escape", "ball_dropped")
        assert info["bounces"] == 0
        assert env.data.qpos[env.ball_q]-q0[0] > .08
        assert np.linalg.norm(env.data.qpos[env.ball_q+3:env.ball_q+7]-q0[3:]) > .1
    finally:
        env.close()


def test_observation_and_baseline_use_camera_estimate_without_truth_fallback(monkeypatch):
    from rebotarm_pingpong.vision import BallEstimate
    env = PingPong1DEnv()
    try:
        env.reset(seed=8)
        fixed = BallEstimate(np.array([.31, .01, .55]), np.array([.02, -.03, -.8]),
                             True, .012, .8, .0, .012, (None, None))
        monkeypatch.setattr(env.observer, "observe", lambda data: fixed)
        def forbidden():
            raise AssertionError("controller consulted ball truth")
        monkeypatch.setattr(env, "state", forbidden)
        monkeypatch.setattr(env, "truth_ball_state", forbidden)
        first = env._observation()
        command = env.baseline_speed
        env.data.qpos[env.ball_q:env.ball_q+3] = [.5, .2, .8]
        env.data.qvel[env.ball_dof:env.ball_dof+6] = [2., -3., 4., 30., 40., 50.]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_array_equal(env._observation(), first)
        assert env.baseline_speed == command
        invalid = BallEstimate(fixed.position, fixed.velocity, False, .2, 0., 0., .012, (None, None))
        monkeypatch.setattr(env.observer, "observe", lambda data: invalid)
        obs = env._observation()
        assert obs[-4] == 0. and np.all(obs[:6] == 0.)
        paddle, _, _ = env.servo.paddle_state(env.data)
        assert env.baseline_speed == pytest.approx(np.clip(8*(env.baseline.home_z-paddle[2]), -.35, .35))
    finally:
        env.close()


def test_missing_camera_frames_terminate_without_truth_recovery():
    env = PingPong1DEnv()
    try:
        env.observer.settings["dropout_probability"] = 1.
        env.reset(seed=3, options={"height_m": .45, "velocity_m_s": .8,
                                  "horizontal_velocity_m_s": [0., 0.], "spin_rad_s": [0., 0., 0.]})
        for _ in range(40):
            obs, _, done, _, info = env.step([.8])
            assert obs[-4] == 0.
            if done:
                break
        assert done and info["termination"] == "vision_lost"
        assert info["vision_valid_fraction"] == 0.
    finally:
        env.close()


def test_drop_rebound_is_passive_and_converges_with_smaller_timestep():
    report = drop_report()
    assert report["within_10_percent"]
    assert all(.3 < t["effective_restitution"] < .9 for t in report["trials"])
    assert all(.0003 < t["contact_s"] < .003 for t in report["trials"])
    assert all(t["max_penetration_m"] > 0 and t["peak_normal_force_n"] > 0 and t["normal_impulse_n_s"] > 0 for t in report["trials"])
    assert report["physical_calibrated"] is False


def test_camera_covers_flight_and_checkpoint_rejects_wrong_model(tmp_path):
    report = camera_report()
    assert report["in_frame_fraction"] == 1.
    assert report["min_ball_diameter_px"] > 10
    fingerprint = model_fingerprint()
    fingerprint["robot_pingpong.xml"] = "wrong"
    (tmp_path/"metadata.json").write_text(json.dumps({"model_fingerprint": fingerprint}))
    with pytest.raises(ValueError, match="changed since training"):
        load_policy(tmp_path/"policy.zip")


def test_randomized_env_seed_is_reproducible_and_yaml_override_applies(tmp_path):
    from stable_baselines3.common.env_checker import check_env
    env = PingPong1DEnv(randomize=True)
    check_env(env)
    a, _ = env.reset(seed=2026)
    first = env.step([.1])[0]
    b, _ = env.reset(seed=2026)
    second = env.step([.1])[0]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first, second)
    config = yaml.safe_load((description_path()/"config/pingpong.yaml").read_text())
    config["ball"]["mass_kg"] = .003
    override = tmp_path/"pingpong.yaml"
    override.write_text(yaml.safe_dump(config))
    other = PingPong1DEnv(config_path=override)
    other.reset(seed=0)
    assert other.model.body_mass[other.ball_body] == pytest.approx(.003)
    env.close()
    other.close()


def test_environment_rejects_unsynchronized_canonical_config(tmp_path, monkeypatch):
    from rebotarm_pingpong import env as module
    source = description_path()
    for relative in ("urdf/robot_arm.urdf", "config/pingpong.yaml", "config/rebotarm_hardware.yaml", "mjcf/robot.xml",
                     "mjcf/robot_pingpong.xml", "mjcf/pingpong_1d_scene.xml", "mjcf/pingpong_3d_scene.xml"):
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, target)
    path = tmp_path/"config/pingpong.yaml"
    c = yaml.safe_load(path.read_text())
    c["ball"]["mass_kg"] = .003
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(module, "description_path", lambda: tmp_path)
    with pytest.raises(ValueError, match="stale"):
        PingPong1DEnv()


def test_viewer_starts_at_overview_but_mouse_drag_changes_free_camera():
    from rebotarm_pingpong.viewer import configure_camera
    env = PingPong1DEnv()
    env.reset(seed=0)
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)

    def eye_position():
        mujoco.mjv_updateCamera(env.model, env.data, camera, scene)
        return (scene.camera[0].pos + scene.camera[1].pos) / 2

    configure_camera(camera, env.model, env.config)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    np.testing.assert_allclose(eye_position(), env.config["cameras"]["overview"]["position_m"], atol=2e-7)
    before = eye_position().copy()
    mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, .1, .1, camera)
    assert np.linalg.norm(eye_position()-before) > .01
    before_distance = camera.distance
    mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., .1, camera)
    assert camera.distance != before_distance
    before_target = camera.lookat.copy()
    mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_MOVE_H, .1, 0., camera)
    assert np.linalg.norm(camera.lookat-before_target) > .01
    for mode in ("overview", "tracking", "tracking_side"):
        configure_camera(camera, env.model, env.config, mode)
        assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
        assert camera.fixedcamid == env.model.camera(f"pingpong_{mode}").id
        before = eye_position().copy()
        mujoco.mjv_moveCamera(env.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, .1, .1, camera)
        np.testing.assert_array_equal(eye_position(), before)
    env.close()
