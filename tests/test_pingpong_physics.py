"""Physics checks independent of policy success or privileged observations.

The isolated drop rig reuses the generated ball/pair/paddle attributes so the
arm servo cannot hide collision energy or affect the contact convergence test.
These checks establish numerical plausibility, not calibration to real rubber.
"""
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from rebotarm_description.parameters import description_path
from rebotarm_description.pingpong_model import validate_config


def _config():
    return yaml.safe_load((description_path() / "config/pingpong.yaml").read_text())


def _drop_rig(height, timestep, tilt=0.0):
    source = description_path() / "mjcf"
    scene = ET.parse(source / "pingpong_3d_scene.xml").getroot()
    robot = ET.parse(source / "robot_pingpong.xml").getroot()
    root = ET.Element("mujoco")
    option = copy.deepcopy(scene.find("option"))
    option.set("timestep", str(timestep))
    root.append(option)
    world = ET.SubElement(root, "worldbody")
    support = ET.SubElement(world, "body")
    support.set("quat", " ".join(map(str, Rotation.from_euler("y", tilt).as_quat()[[3, 0, 1, 2]])))
    paddle = copy.deepcopy(robot.find(".//geom[@name='paddle_face']"))
    support.append(paddle)
    ball = copy.deepcopy(scene.find(".//body[@name='pingpong_ball']"))
    paddle_half = float(paddle.get("size").split()[1])
    radius = float(ball.find("geom").get("size"))
    ball.set("pos", f"0 0 {height + radius + paddle_half}")
    world.append(ball)
    root.append(copy.deepcopy(scene.find("contact")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def _drop(height, timestep, tilt=0.0):
    model, data = _drop_rig(height, timestep, tilt)
    prior_contact = False
    penetration = 0.0
    for _ in range(round(0.8 / timestep)):
        before_vz = data.qvel[2]
        mujoco.mj_step(model, data)
        contact = data.ncon > 0
        for pair in data.contact:
            penetration = max(penetration, -float(pair.dist))
        if contact and not prior_contact:
            incoming = before_vz
            start = data.time
        if prior_contact and not contact:
            return {"e": data.qvel[2] / -incoming,
                    "contact_s": data.time - start,
                    "penetration_m": penetration,
                    "outgoing_velocity": data.qvel.copy()}
        prior_contact = contact
    raise AssertionError("Drop rig did not produce a separated impact")


@pytest.mark.parametrize("scene", ["pingpong_1d_scene.xml", "pingpong_3d_scene.xml"])
def test_both_ball_scenes_translate_rotate_and_use_hollow_shell_inertia(scene):
    c = _config()
    model = mujoco.MjModel.from_xml_path(str(description_path() / "mjcf" / scene))
    joint = model.joint("pingpong_ball_freejoint")
    assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    inertia = (2 / 3) * c["ball"]["mass_kg"] * c["ball"]["radius_m"] ** 2
    np.testing.assert_allclose(model.body("pingpong_ball").inertia, [inertia] * 3, rtol=1e-12)
    data = mujoco.MjData(model)
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[qadr:qadr + 3] = [1.2, -1.2, 1.3]
    data.qvel[dadr:dadr + 6] = [.25, -.18, -.15, 2.0, -3.0, 4.0]
    before = data.qpos[qadr:qadr + 7].copy()
    steps = 80
    for _ in range(steps):
        mujoco.mj_step(model, data)
    duration = steps * model.opt.timestep
    np.testing.assert_allclose(data.qpos[qadr:qadr + 2] - before[:2], duration * np.array([.25, -.18]), atol=1e-12)
    assert np.linalg.norm(data.qpos[qadr + 3:qadr + 7] - before[3:7]) > .01
    assert data.qvel[dadr + 2] == pytest.approx(-.15 - c["scene"]["gravity_m_s2"] * duration)


def test_short_contact_prior_is_passive_resolved_and_converges():
    c = _config()
    dt = c["scene"]["timestep_s"]
    for height in c["calibration"]["drop_heights_m"]:
        nominal = _drop(height, dt)
        refined = _drop(height, dt / 4)
        assert 0.5 < nominal["e"] < 0.95
        assert abs(nominal["e"] - refined["e"]) / refined["e"] < .10
        assert .0005 <= nominal["contact_s"] <= .002
        assert 0 < nominal["penetration_m"] < .001
        assert nominal["contact_s"] / dt >= 5
    assert c["calibration"]["status"] == "unmeasured"


def test_tilted_paddle_collision_produces_sideways_ball_motion():
    result = _drop(.15, _config()["scene"]["timestep_s"], tilt=.05)
    # A slide-joint ball would absorb this collision impulse in its constraint.
    assert result["outgoing_velocity"][0] > .04


def test_calibration_label_requires_real_measurement_fields_and_passivity():
    c = _config()
    c["calibration"]["status"] = "measured"
    with pytest.raises(ValueError, match="three real drop"):
        validate_config(c)
    c["calibration"].update(ball_id="recorded-ball", paddle_id="recorded-rubber")
    c["calibration"]["measurements"] = [dict(drop_height_m=.15, rebound_height_m=.08,
         contact_duration_s=.001, incoming_speed_m_s=1.7, outgoing_speed_m_s=1.3)] * 3
    validate_config(c)
    c["calibration"]["measurements"][0]["outgoing_speed_m_s"] = 2.0
    with pytest.raises(ValueError, match="cannot add energy"):
        validate_config(c)


def test_drop_audit_has_no_renderer_or_servo_and_checks_real_data_fit(monkeypatch):
    from rebotarm_pingpong.diagnostics import drop_report, _model_sources, _calibration_comparison

    def forbid_renderer(*args, **kwargs):
        raise AssertionError("A fixed-paddle physical audit must not allocate a visual observer")

    monkeypatch.setattr(mujoco, "Renderer", forbid_renderer)
    report = drop_report()
    assert report["within_10_percent"]
    assert report["passive_energy_checks_passed"]
    assert report["off_paddle_trial"]["impact"] is False
    assert report["off_paddle_trial"]["final_position_m"][2] < -.5
    assert report["physical_calibrated"] is False
    assert report["calibration_comparison"]["measurements_count"] == 0
    assert report["max_penetration_m"] < .001
    assert len(report["stress_trials"]) >= 3
    assert np.linalg.norm(report["stress_trials"][-1]["outgoing_velocity_m_s"][:2]) > .05

    _, c, files = _model_sources()
    # Synthetic samples verify the fitting gate only. They are not stored or
    # represented as hardware measurements in the repository configuration.
    c["calibration"].update(status="measured", ball_id="synthetic-test-only", paddle_id="synthetic-test-only")
    c["calibration"]["measurements"] = [dict(
        drop_height_m=t["height_m"], rebound_height_m=t["rebound_height_m"],
        contact_duration_s=t["contact_s"], incoming_speed_m_s=-t["incoming_relative_m_s"],
        outgoing_speed_m_s=t["outgoing_relative_m_s"])
        for t in report["trials"][::3]]
    assert _calibration_comparison(c, files)["fit_within_tolerance"]
    c["calibration"]["measurements"][0]["contact_duration_s"] *= 3
    comparison = _calibration_comparison(c, files)
    assert not comparison["fit_within_tolerance"]
    assert comparison["comparisons"][0]["restitution_abs_error"] < .001
    assert comparison["comparisons"][0]["contact_duration_relative_error"] > .5
