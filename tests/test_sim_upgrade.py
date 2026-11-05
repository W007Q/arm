from pathlib import Path
import numpy as np
import pytest
import mujoco
from rebotarm_vision.rgbd_detection import detect_blocks
from rebotarm_planning.collision_planner import CollisionPlanner, PlanningError

SCENE = Path(__file__).resolve().parents[1]/'src/rebotarm_description/mjcf/grasp_scene.xml'

def render_detections(model, data):
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera='overhead_rgb')
        rgb=renderer.render().copy()
        renderer.enable_depth_rendering()
        depth=renderer.render().copy()
    camera=model.camera('overhead_rgb').id
    f=120/np.tan(np.deg2rad(model.cam_fovy[camera])/2)
    k=np.array([[f,0,159.5],[0,f,119.5],[0,0,1.]])
    rotation=data.cam_xmat[camera].reshape(3,3)@np.diag([1.,-1.,-1.])
    return detect_blocks(rgb,depth,k,rotation,data.cam_xpos[camera])

def test_camera_localizes_and_tracks_moved_object():
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    before={d['color']:d for d in render_detections(model,data)}
    assert set(before)=={'red','blue','yellow'}
    for color,name in [('red','red_cube'),('blue','blue_block'),('yellow','yellow_cylinder')]:
        found=before[color];actual=data.body(name).xpos
        assert np.linalg.norm(np.array([found['x'],found['y']])-actual[:2]) < .008
        assert found['source']=='camera_rgbd'
    address=model.joint('red_cube_freejoint').qposadr[0]
    data.qpos[address]+=.025;data.qpos[address+2]+=.06
    after={d['color']:d for d in render_detections(model,data)}
    assert abs(after['red']['x']-before['red']['x']-.025)<.008
    assert abs(after['red']['z']-before['red']['z']-.06)<.008

def test_base_clears_floor_and_contact_pads_follow_visible_fingertips():
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    floor_z=float(model.geom('floor').pos[2])
    base=model.geom('base_link_mesh_1_collision').id
    mesh=int(model.geom_dataid[base]);address=int(model.mesh_vertadr[mesh])
    count=int(model.mesh_vertnum[mesh]);vertices=model.mesh_vert[address:address+count]
    world=vertices@data.geom_xmat[base].reshape(3,3).T+data.geom_xpos[base]
    assert float(world[:,2].min())-floor_z == pytest.approx(.002,abs=2e-5)
    # Concave STL meshes collide as over-sized convex hulls in MuJoCo.  The
    # visible meshes stay non-colliding and thin boxes model the real pads.
    assert model.geom('gripper_left_mesh_1').contype == 0
    assert model.geom('gripper_right_mesh_1').contype == 0
    assert model.geom('gripper_left_contact').contype == 1
    assert model.geom('gripper_right_contact').contype == 1

    # At a 45 mm test object opening the proxy faces and visible fingertips
    # agree within 2 mm, so an invisible proxy cannot lift the object early.
    grasp_q = [-.3361725208, -1.9135142716, 1.6828348613,
               -1.3430193480, .0010314896, -1.9122762675]
    for index, value in enumerate(grasp_q, 1):
        data.qpos[model.joint(f'joint{index}').qposadr[0]] = value
    slide = .012
    data.qpos[model.joint('joint7').qposadr[0]] = slide
    data.qpos[model.joint('gripper_joint1').qposadr[0]] = -slide
    data.qpos[model.joint('gripper_joint2').qposadr[0]] = slide
    mujoco.mj_forward(model, data)
    contacts = set()
    for contact in data.contact:
        names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
        if 'red_cube_geom' not in names:
            continue
        other = next(name for name in names if name != 'red_cube_geom')
        contacts.add(other)
    assert contacts == {'gripper_left_contact', 'gripper_right_contact'}

def test_d435_is_attached_to_end_effector():
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    camera=model.camera('d435_color').id
    d435=model.body('d435_link').id
    assert int(model.cam_bodyid[camera]) == d435
    before=data.cam_xpos[camera].copy()
    data.qpos[model.joint('joint1').qposadr[0]]=.25
    mujoco.mj_forward(model,data)
    assert np.linalg.norm(data.cam_xpos[camera]-before)>.01

def test_empty_depth_never_produces_target():
    rgb=np.full((50,50,3),[255,0,0],np.uint8)
    assert detect_blocks(rgb,np.zeros((50,50)),np.eye(3),np.eye(3),np.zeros(3))==[]

def test_real_model_collision_rejects_bad_goal():
    planner=CollisionPlanner(SCENE)
    start=planner.data.qpos[planner.addr].copy()
    assert planner.valid(start),planner.last_collision
    bad=np.full(6,np.nan)
    with pytest.raises(PlanningError): planner.plan(start,bad)
    rng=np.random.default_rng(4)
    collision=None
    for _ in range(200):
        q=rng.uniform(planner.lower,planner.upper)
        if not planner.valid(q): collision=q;break
    assert collision is not None
    with pytest.raises(PlanningError):planner.plan(start,collision)

def test_rrt_routes_around_obstacle():
    # Exercise search independently of robot geometry with a wall in 2-D.
    planner=object.__new__(CollisionPlanner)
    planner.lower=np.array([-1.,-1.]);planner.upper=np.array([1.,1.])
    planner.resolution=.025;planner.last_collision='wall'
    planner.valid=lambda q: not(abs(q[0])<.18 and abs(q[1])<.55)
    start=np.array([-.8,0.]);goal=np.array([.8,0.])
    assert not planner.edge(start,goal)
    path=planner.plan(start,goal,timeout=3.)
    assert len(path)>2
    assert all(planner.edge(a,b) for a,b in zip(path,path[1:]))
