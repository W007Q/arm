"""MoveIt must send the trajectory to the selected device namespace."""
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'src/rebotarm_moveit_config'
spec = importlib.util.spec_from_file_location('moveit_launch_common', CONFIG / 'launch/moveit_launch_common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


@pytest.mark.parametrize('namespace', ['rebotarm', 'arm_test', '/cell/arm/'])
def test_hardware_moveit_controller_matches_backend_namespace(namespace):
    original = yaml.safe_load((CONFIG / 'config/moveit_hardware_controllers.yaml').read_text())
    params = common.hardware_controller_namespace(original, namespace)
    manager = params['moveit_simple_controller_manager']
    name = namespace.strip('/')
    assert manager['controller_names'] == [name]
    assert manager[name]['action_ns'] == 'follow_joint_trajectory'
    assert manager[name]['type'] == 'FollowJointTrajectory'
    assert manager[name]['joints'] == [f'joint{i}' for i in range(1, 7)]
    assert original['moveit_simple_controller_manager']['controller_names'] == ['rebotarm']


def test_hardware_controller_rejects_empty_namespace():
    params = yaml.safe_load((CONFIG / 'config/moveit_hardware_controllers.yaml').read_text())
    with pytest.raises(ValueError, match='must not be empty'):
        common.hardware_controller_namespace(params, '/')


def test_optional_jtc_controller_address_is_explicit_and_leaves_default_unchanged():
    params = yaml.safe_load((CONFIG / 'config/moveit_hardware_controllers.yaml').read_text())
    manager = common.hardware_controller_namespace(params, 'cell/arm', 'arm_controller')[
        'moveit_simple_controller_manager']
    assert manager['controller_names'] == ['cell/arm/arm_controller']
    assert manager['cell/arm/arm_controller']['action_ns'] == 'follow_joint_trajectory'
    with pytest.raises(ValueError, match='trajectory_controller'):
        common.hardware_controller_namespace(params, 'cell/arm', 'unsupported')


@pytest.mark.parametrize('controller,monitoring', [('', False), ('arm_controller', True)])
def test_execution_monitor_matches_selected_backend(controller, monitoring):
    original = yaml.safe_load((CONFIG / 'config/moveit_hardware_controllers.yaml').read_text())
    params = common.hardware_controller_namespace(original, 'rebotarm', controller)
    # A feedback-paced backend can exceed the planned duration. MoveIt must wait
    # for its result; JTC keeps the fixed-duration monitor from the source YAML.
    assert params['trajectory_execution']['execution_duration_monitoring'] is monitoring
    assert original['trajectory_execution']['execution_duration_monitoring'] is True
    assert params['trajectory_execution']['allowed_start_tolerance'] == .05
