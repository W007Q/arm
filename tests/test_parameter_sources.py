"""Guard propagation of model and motor changes across the real backends."""
import copy
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from rebotarm_description.parameters import ARM_JOINTS, arm_gains, joint_limits, load_hardware, load_mounts, uart_gains, validate
from rebotarmcontroller.hardware_config import _add_runtime_config

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ROOT / 'src/rebotarm_description'
spec = importlib.util.spec_from_file_location('parameter_sync', ROOT / 'scripts/sync_robot_parameters.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_checked_in_files_have_no_parameter_drift():
    for path, expected in sync.derived_files(ROOT).items():
        assert path.read_text() == expected, f'{path}: run scripts/sync_robot_parameters.py'
    limits = joint_limits()
    model = mujoco.MjModel.from_xml_path(str(DESCRIPTION / 'mjcf/grasp_scene.xml'))
    bridge = yaml.safe_load((ROOT / 'src/rebotarm_uart/config/uart_grasp.yaml').read_text())
    safety = bridge['rebotarm_uart_bridge']['ros__parameters']
    adapter = bridge['rebotarm_posvel_to_mit']['ros__parameters']
    assert safety['slot_joint_ids'] == [4, 6, 1, 2, 3, 5, 7]
    assert safety['slot_feedback_ids'] == [4, 5, 17, 18, 19, 6, 7]
    assert safety['feedback_zero_tolerance_rad'] == [0., 0., 0., 0., .01, 0.]
    assert safety['torque_limit'] == [1.0, 1.0, 1.0, 7.0, 3.0, 7.0, 3.0]
    for i, name in enumerate(ARM_JOINTS):
        np.testing.assert_allclose(model.joint(name).range, [limits[name]['lower'], limits[name]['upper']])
        assert [safety['position_min'][i], safety['position_max'][i]] == list(model.joint(name).range)
    # Exercise the SDK's actual runtime gain extraction, without creating a bus.
    sdk = copy.deepcopy(load_hardware()['models']['hybrid']['overrides'])
    _add_runtime_config(sdk)
    # SDK/CAN and STM32 UART use the same logical-joint MIT source.
    assert sdk['_runtime']['control']['mit_kp'] == arm_gains(load_hardware())['kp']
    assert sdk['_runtime']['control']['mit_kd'] == arm_gains(load_hardware())['kd']
    assert adapter['arm_kp'] == uart_gains(load_hardware())['kp'][:6]
    assert adapter['arm_kd'] == uart_gains(load_hardware())['kd'][:6]


def test_source_changes_reach_compiled_model_uart_and_sdk(tmp_path):
    desc = tmp_path / 'src/rebotarm_description'
    for relative in ('urdf/robot_arm.urdf', 'config/rebotarm_hardware.yaml', 'config/d435_mount.yaml', 'config/pingpong.yaml', 'mjcf/robot.xml', 'mjcf/grasp_scene.xml'):
        target = desc / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DESCRIPTION / relative, target)
    (desc / 'meshes').symlink_to(DESCRIPTION / 'meshes', target_is_directory=True)
    urdf = ET.parse(desc / 'urdf/robot_arm.urdf')
    j4 = urdf.find("joint[@name='joint4']/limit")
    j4.set('lower', '-1.2'); j4.set('upper', '1.2')
    urdf.find("link[@name='link2']/inertial/mass").set('value', '0.9')
    urdf.write(desc / 'urdf/robot_arm.urdf')
    hardware = load_hardware(desc / 'config/rebotarm_hardware.yaml')
    profile = hardware['models']['hybrid']['overrides']
    profile['joints'][3]['motor_id'] = 44
    hardware['uart']['transport']['feedback_motor_ids'][0] = 44
    hardware['motor_catalog']['4310']['peak'] = 6.
    # A single logical-joint MIT edit propagates to SDK and UART; UART slot
    # permutation is applied only while encoding the fixed wire frame.
    profile['joints'][3]['MIT']['kp'] = 2.4
    (desc / 'config/rebotarm_hardware.yaml').write_text(yaml.safe_dump(hardware))
    mounts = load_mounts(desc)
    mounts['tcp']['xyz_m'][0] = .012
    mounts['d435_mount']['xyz_m'][0] = -.07
    (desc / 'config/d435_mount.yaml').write_text(yaml.safe_dump(mounts))
    outputs = sync.derived_files(tmp_path)
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    model = mujoco.MjModel.from_xml_path(str(desc / 'mjcf/grasp_scene.xml'))
    np.testing.assert_allclose(model.joint('joint4').range, [-1.2, 1.2])
    assert model.body('link2').mass[0] == pytest.approx(.9)
    assert model.body('d435_link').pos[0] == pytest.approx(-.07)
    assert model.site('gripper_tcp').pos[0] == pytest.approx(.012)
    for filename in ('uart.yaml', 'uart_grasp.yaml'):
        params = yaml.safe_load((tmp_path / 'src/rebotarm_uart/config' / filename).read_text())
        bridge = params['rebotarm_uart_bridge']['ros__parameters']
        assert bridge['position_min'][3] == -1.2 and bridge['position_max'][3] == 1.2
        # CAN ID 由 STM32 映射；UART 槽位始终是逻辑 ID 1..7。
        assert bridge['joint_ids'] == list(range(1, 8)) and bridge['torque_limit'][3] == 6.
        assert params['rebotarm_posvel_to_mit']['ros__parameters']['arm_kp'][3] == 2.4
    sdk = copy.deepcopy(profile)
    _add_runtime_config(sdk)
    assert sdk['_runtime']['control']['mit_kp'][3] == 2.4
    specs = yaml.safe_load((desc / 'config/motor_specs.yaml').read_text())
    assert specs['joints']['joint4']['peak'] == 6.
    assert specs['payload']['tcp_offset_m'][0] == .012
    # Regeneration is idempotent even after source changes.
    assert outputs == sync.derived_files(tmp_path)


def test_invalid_source_fails_before_any_generation():
    limits = joint_limits()
    limits['joint4']['lower'] = 2.
    with pytest.raises(ValueError, match='joint4'):
        validate(load_hardware(), limits, load_mounts())


@pytest.mark.parametrize('opened,closed', [(5., .4), (-5., 0.)])
def test_motor_calibration_handles_shifted_zero_and_reversed_direction(monkeypatch, opened, closed):
    from rebotarm_mujoco_rs import mujoco_sync
    monkeypatch.setattr(mujoco_sync, '_GRIPPER_MOTOR_OPEN_RAD', opened)
    monkeypatch.setattr(mujoco_sync, '_GRIPPER_MOTOR_CLOSE_RAD', closed)
    node = SimpleNamespace(master_qpos_addr=0, data=SimpleNamespace(qpos=np.zeros(1)))
    for fraction in (0., .3, 1.):
        target = closed + fraction * (opened - closed)
        mujoco_sync.RsMujocoSync._set_gripper_target_rad(node, target)
        assert node.target_gripper == pytest.approx(fraction * mujoco_sync._MUJOCO_GRIPPER_OPEN_M)
        node.data.qpos[0] = node.target_gripper
        assert mujoco_sync.RsMujocoSync._current_gripper_rad(node) == pytest.approx(target)
