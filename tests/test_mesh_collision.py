"""现场姿态回归：区分凸包误报、CAD 相交和外部障碍。"""
from pathlib import Path

import numpy as np
import pytest

from rebotarm_planning.collision_planner import CollisionPlanner
from rebotarm_planning.scene import HardwareScene


SCENE = Path(__file__).resolve().parents[1] / 'src/rebotarm_description/mjcf/hardware_scene.xml'


@pytest.fixture(scope='module')
def planner():
    p = CollisionPlanner(SCENE, exact_meshes=True)
    HardwareScene(p, table_top_m=-.0744, camera_collision=False)
    return p


def test_convex_hull_false_contact_does_not_block_clear_cad_path(planner):
    # New CAD (2026-10-26): both hulls overlap but their triangle surfaces clear.
    q = np.array([0., 0., 0., 0., -.007, 0.])
    assert planner.valid(q)
    # MuJoCo 凸包仍判穿透；不是靠删除候选接触或放宽容差来通过。
    assert any(c.dist < -.00005 and
               {planner.model.body(int(planner.model.geom_bodyid[g])).name
                for g in (c.geom1, c.geom2)} == {'link1', 'link5'}
               for c in planner.data.contact)
    goal = q.copy()
    goal[2] = .1
    assert planner.edge(q, goal)


def test_confirmed_link2_link5_stop_allows_recorded_zero_drift(planner):
    # 2026-10-24 现场反馈，J3 边界量化值按执行端规则夹到零。
    q = np.array([-.0001907378, -.0001907378, 0., -.0040054931,
                  -.0005722133, -.0001907378])
    # 用户已确认该连杆对为机械限位；保留实测偏差也允许离开限位。
    assert planner.valid(q)
    q[3] = -.0001907378
    q[4] = .0074387733
    goal = q.copy()
    goal[2] = .12
    assert planner.edge(q, goal)
    np.testing.assert_allclose(planner.data.qpos[planner.addr], goal)


def test_exact_mesh_check_preserves_deep_self_and_external_collision_guards(planner):
    q = np.array([1.0887, -1.1648, .088, -.6404, 1.342, 1.788])
    assert not planner.valid(q)
    assert 'CAD triangle meshes intersect' in planner.last_collision
    assert planner.valid(np.zeros(6))
    adr = int(planner.model.joint('red_cube_freejoint').qposadr[0])
    planner.data.qpos[adr:adr+3] = planner.data.body('link4').xpos
    try:
        assert not planner.valid(np.zeros(6))
        assert 'red_cube' in planner.last_collision
    finally:
        planner.data.qpos[adr:adr+3] = [0., 0., -10.]


def test_camera_enabled_after_mesh_cache_creation_still_checks_its_surface():
    from rebotarm_planning.mesh_collision import MeshCollisionCheck
    p = CollisionPlanner(SCENE)
    HardwareScene(p, table_top_m=-.042275, camera_collision=False)
    # Simulate startup with the camera disabled, followed by enabling it.
    p.mesh_check = MeshCollisionCheck(p.model, p.robot)
    HardwareScene(p, table_top_m=-.042275, camera_collision=True)
    q = np.array([-.28263676, -1.37175053, .40000443,
                  -1.21417547, -.61195132, 2.65225184])
    assert not p.valid(q)
    camera = p.model.geom('d435_body_visual').id
    base_arm = p.model.geom('link1_mesh_1_collision').id
    assert p.mesh_check.collides(p.data, camera, base_arm)
