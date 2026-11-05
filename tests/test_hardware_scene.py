from pathlib import Path

import mujoco
import numpy as np
import pytest

from rebotarm_planning.collision_planner import CollisionPlanner, PlanningError
from rebotarm_planning.scene import HardwareScene

SCENE = Path(__file__).resolve().parents[1] / 'src/rebotarm_description/mjcf/hardware_scene.xml'


def make_scene():
    planner = CollisionPlanner(SCENE)
    return planner, HardwareScene(planner, table_top_m=.055725, ttl=10.)


def detection(source='camera_rgbd'):
    return {'detections': [{'color': 'red', 'source': source, 'x': .34, 'y': -.12,
                            'z': .10, 'shortest_m': .045, 'longest_m': .08}]}


def test_zero_pose_can_leave_mechanical_stop():
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744, camera_collision=False)
    start = np.zeros(6)
    scene.prepare('', start, {})
    goal = start.copy()
    goal[2] = .10
    path = planner.plan(start, goal)
    assert all(planner.edge(a, b) for a, b in zip(path, path[1:]))
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)


def test_measured_zero_with_encoder_quantization_can_leave_stop():
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744, camera_collision=False)
    # 现场反馈；J3 的微小负数与 task_server 一样夹到关节下限。
    start = np.array([-.00019073777366429567] * 6)
    start[2] = 0.
    start[4] = -.000572213320992887
    scene.prepare('', start, {})
    assert planner.valid(start)
    # 保留实测量化偏差；新 CAD 此姿态有间隙，不要求制造旧模型的浅穿透。
    np.testing.assert_array_equal(planner.data.qpos[planner.addr], start)
    goal = start.copy()
    goal[2] = .10
    path = planner.plan(start, goal)
    assert all(planner.edge(a, b) for a, b in zip(path, path[1:]))
    np.testing.assert_array_equal(path[0], start)


def test_contact_tolerance_still_rejects_wrist_penetration_near_zero():
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744, camera_collision=False)
    q = np.zeros(6)
    q[4] = .02  # link1/link5 穿透约 0.31 mm，超过数值容差。
    scene.prepare('', q, {})
    assert not planner.valid(q)
    assert 'link1 / link5' in planner.last_collision
    assert 'distance=' in planner.last_collision


def test_mechanical_stop_allowance_still_rejects_other_self_collisions():
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744, camera_collision=False)
    # This folded wrist penetrates link1/link5, away from the allowed stop.
    q = np.array([1.0887, -1.1648, .0880, -.6404, 1.3420, 1.7880])
    scene.prepare('', q, {})
    with pytest.raises(PlanningError, match='collision'):
        planner.plan(np.zeros(6), q)


def test_zero_pose_still_rejects_external_obstacle():
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744, camera_collision=False)
    scene.prepare('', np.zeros(6), {})
    assert planner.valid(np.zeros(6))
    # Place a measured-object proxy inside the arm, rather than excluding it.
    adr = int(planner.model.joint('red_cube_freejoint').qposadr[0])
    planner.data.qpos[adr:adr+3] = planner.data.body('link4').xpos
    assert not planner.valid(np.zeros(6))
    assert 'red_cube' in planner.last_collision


@pytest.mark.parametrize('camera_collision', [False, True])
def test_installed_camera_proxy_is_checked_only_when_present(camera_collision):
    planner = CollisionPlanner(SCENE)
    scene = HardwareScene(planner, table_top_m=-.0744,
                          camera_collision=camera_collision)
    scene.prepare('', np.zeros(6), {})
    # The imported housing clears the arm at zero, with collision enabled.
    assert planner.valid(np.zeros(6))
    # Put an external box over the camera top, away from the wrist. This checks
    # the runtime broadphase masks, not just a flag in the source XML.
    adr = int(planner.model.joint('red_cube_freejoint').qposadr[0])
    planner.data.qpos[adr:adr+3] = [.158387, -.00053739, .21632499]
    assert planner.valid(np.zeros(6)) is (not camera_collision)
    if camera_collision:
        assert 'camera_link' in planner.last_collision


def test_hardware_scene_requires_real_measurement_and_fresh_selected_target():
    planner, scene = make_scene()
    q = np.zeros(6)
    scene.update(detection('mujoco_rgbd'), now=1.)
    assert not scene.objects
    with pytest.raises(PlanningError, match='missing or stale'):
        scene.prepare('red', q, {}, now=2.)
    # Observing from a static, calibrated fixture map does not need a target yet.
    scene.prepare('', q, {}, now=2.)
    scene.update(detection(), now=3.)
    scene.prepare('red', q, {}, now=4.)
    with pytest.raises(PlanningError, match='missing or stale'):
        scene.prepare('red', q, {}, now=14.)


def test_measured_geometry_refreshes_bounds_and_preserves_arm_configuration():
    planner, scene = make_scene()
    planner.data.qpos[planner.addr] = [.2, -1., 1., -.5, .1, .3]
    before = planner.data.qpos.copy()
    scene.update(detection())
    np.testing.assert_array_equal(planner.data.qpos, before)
    gid = planner.model.geom('red_cube_geom').id
    np.testing.assert_allclose(planner.model.geom_size[gid], [.04, .04, (.10-.055725)/2])
    assert planner.model.geom_rbound[gid] > .04
    # A blue proxy 60 mm away overlaps the new 80 mm red box, but did not
    # overlap its original 45 mm box. Exercise collision broadphase as well.
    blue = planner.model.joint('blue_block_freejoint').qposadr[0]
    planner.data.qpos[blue:blue+7] = [.40, -.12, .08, 1., 0., 0., 0.]
    scene.prepare('red', planner.data.qpos[planner.addr].copy(), {})
    mujoco.mj_forward(planner.model, planner.data)
    blue_geom = planner.model.geom('blue_block_geom').id
    assert any({c.geom1,c.geom2} == {gid,blue_geom} for c in planner.data.contact)


def test_attachment_rejects_unmeasured_objects_and_does_not_grow_with_lift():
    planner, scene = make_scene()
    with pytest.raises(PlanningError, match='without measured'):
        scene.attach('red', np.zeros(6))
    scene.update(detection())
    scene.attach('red', np.zeros(6))
    before = planner.model.geom_size[planner.model.geom('red_cube_geom').id].copy()
    lifted = detection(); lifted['detections'][0]['z'] += .1
    scene.update(lifted)
    np.testing.assert_array_equal(planner.model.geom_size[planner.model.geom('red_cube_geom').id], before)
    scene.attach('release', np.zeros(6))
    assert scene.attached is None
