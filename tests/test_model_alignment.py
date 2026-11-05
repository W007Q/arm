"""Catch CAD/world frame changes before a simulated target reaches hardware."""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rebotarm_planning.sdk_model import SdkRobotModel
from rebotarm_description.parameters import load_mounts, uart_parameters

DESCRIPTION = Path(__file__).resolve().parents[1] / 'src/rebotarm_description'


def test_cad_mesh_vertices_keep_the_same_world_pose_in_mujoco():
    """FK alone misses a wrong mesh RPY convention; compare actual vertices."""
    pin = pytest.importorskip('pinocchio')
    urdf = ET.parse(DESCRIPTION / 'urdf/robot_arm.urdf').getroot()
    robot = pin.buildModelFromUrdf(str(DESCRIPTION / 'urdf/robot_arm.urdf'))
    state = robot.createData()
    model = mujoco.MjModel.from_xml_path(str(DESCRIPTION / 'mjcf/robot.xml'))
    data = mujoco.MjData(model)
    for q in (np.zeros(6), np.array([.2, -.4, .5, -.2, -.1, .3])):
        full = pin.neutral(robot)
        full[:6] = q
        pin.forwardKinematics(robot, state, full)
        pin.updateFramePlacements(robot, state)
        for index, value in enumerate(q, 1):
            data.qpos[model.joint(f'joint{index}').qposadr[0]] = value
        mujoco.mj_forward(model, data)
        for link in urdf.findall('link'):
            pose = state.oMf[robot.getFrameId(link.get('name'))]
            for index, visual in enumerate(link.findall('visual'), 1):
                mesh = visual.find('geometry/mesh')
                payload = (DESCRIPTION / 'meshes' / Path(mesh.get('filename')).name).read_bytes()
                count = struct.unpack_from('<I', payload, 80)[0]
                vertices = np.ndarray((count, 3, 3), dtype='<f4', buffer=payload,
                                      offset=96, strides=(50, 12, 4)).reshape(-1, 3)
                vertices = vertices * np.array(mesh.get('scale').split(), dtype=float)
                origin = visual.find('origin')
                rot = Rotation.from_euler('xyz', np.array(origin.get('rpy').split(), dtype=float))
                vertices = rot.apply(vertices) + np.array(origin.get('xyz').split(), dtype=float)
                expected = vertices @ pose.rotation.T + pose.translation
                prefix = 'base_link' if link.get('name') == 'cad_base_link' else link.get('name')
                gid = model.geom(f'{prefix}_mesh_{index}').id
                mid = model.geom_dataid[gid]
                start, size = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                actual = model.mesh_vert[start:start+size] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                assert cKDTree(actual).query(expected)[0].max() < 1e-6, (link.get('name'), index)


def test_installed_moveit_meshes_exist_and_limits_match_uart():
    import xacro
    from ament_index_python.packages import get_package_share_directory
    xacro_path = Path(get_package_share_directory('rebotarm_moveit_config')) / 'config/rebotarm_hybrid.urdf.xacro'
    root = ET.fromstring(xacro.process_file(str(xacro_path), mappings={'d435_collision': 'false'}).toxml())
    for mesh in root.findall('.//mesh'):
        uri = mesh.get('filename')
        assert uri.startswith('package://'), uri
        package, relative = uri.removeprefix('package://').split('/', 1)
        path = Path(get_package_share_directory(package)) / relative
        assert path.is_file(), f'RViz mesh missing: {path}'
    limits = uart_parameters()
    for index, name in enumerate(limits['joint_names'][:6]):
        joint = root.find(f"joint[@name='{name}']/limit")
        np.testing.assert_allclose([float(joint.get('lower')), float(joint.get('upper'))],
                                   [limits['position_min'][index], limits['position_max'][index]])


def test_urdf_and_mujoco_tcp_camera_and_gravity_agree():
    pytest.importorskip('pinocchio')
    model = mujoco.MjModel.from_xml_path(str(DESCRIPTION / 'mjcf/grasp_scene.xml'))
    data = mujoco.MjData(model)
    tcp = SdkRobotModel(DESCRIPTION / 'urdf/rebotarm_with_d435.urdf.xacro', 'gripper_tcp')
    camera = SdkRobotModel(DESCRIPTION / 'urdf/rebotarm_with_d435.urdf.xacro', 'camera_color_optical_frame')
    joints = [model.joint(f'joint{i}') for i in range(1, 7)]
    ranges = np.array([j.range for j in joints])
    np.testing.assert_allclose(tcp.joint_limits(), ranges, atol=1e-7)
    rng = np.random.default_rng(2026)
    poses = [np.zeros(6), np.array([-.336, -1.913, 1.683, -1.343, .001, -1.912])]
    poses.extend(rng.uniform(ranges[:, 0], ranges[:, 1], size=(20, 6)))
    for q in poses:
        for joint, value in zip(joints, q):
            data.qpos[joint.qposadr[0]] = value
        mujoco.mj_forward(model, data)
        hand = model.body('gripper_end').id
        rotation = data.xmat[hand].reshape(3, 3)
        position = data.xpos[hand] + rotation @ np.array(load_mounts()['tcp']['xyz_m'])
        urdf_position, urdf_rotation = tcp.fk(q)
        np.testing.assert_allclose(position, urdf_position, atol=2e-7)
        np.testing.assert_allclose(rotation, urdf_rotation, atol=2e-7)
        cid = model.camera('d435_color').id
        urdf_position, urdf_rotation = camera.fk(q)
        np.testing.assert_allclose(data.cam_xpos[cid], urdf_position, atol=2e-7)
        np.testing.assert_allclose(data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.]), urdf_rotation, atol=2e-7)
        # MJCF adds a small slide mechanism mass; allow 0.01 Nm for that proxy.
        np.testing.assert_allclose(data.qfrc_bias[[j.dofadr[0] for j in joints]], tcp.gravity(q), atol=.01)
