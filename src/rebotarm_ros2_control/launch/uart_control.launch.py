"""Opt-in ros2_control layer; existing UART nodes remain the serial owners."""
from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rebotarm_description.parameters import uart_parameters


def robot_description(namespace, max_joint_speed=.2, d435_collision='false'):
    """Add the six-axis interface to the canonical model, without copied limits."""
    description = Path(get_package_share_directory('rebotarm_description'))
    root = ET.fromstring(xacro.process_file(
        str(description / 'urdf/rebotarm_with_d435.urdf.xacro'),
        mappings={'d435_internal_tf': 'false', 'd435_collision': d435_collision},
    ).toxml())
    physical = uart_parameters()
    control = ET.SubElement(root, 'ros2_control', name='RebotArmUart', type='system')
    hardware = ET.SubElement(control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'rebotarm_ros2_control/UartSystem'
    values = {'arm_namespace': namespace, 'command_hz': physical['tx_hz'],
              'feedback_timeout_sec': physical['feedback_timeout'], 'command_timeout_sec': .2}
    for key, value in values.items():
        ET.SubElement(hardware, 'param', name=key).text = str(value)
    for index in range(6):
        name = f'joint{index+1}'
        joint = ET.SubElement(control, 'joint', name=name)
        command = ET.SubElement(joint, 'command_interface', name='position')
        limit = root.find(f"joint[@name='{name}']/limit")
        for key, source in [('min', 'lower'), ('max', 'upper')]:
            ET.SubElement(command, 'param', name=key).text = limit.get(source)
        for interface in ('position', 'velocity', 'effort'):
            ET.SubElement(joint, 'state_interface', name=interface)
        ET.SubElement(joint, 'param', name='velocity_limit').text = str(
            min(max_joint_speed, physical['velocity_limit'][index]))
    return ET.tostring(root, encoding='unicode')


def _setup(context):
    namespace = LaunchConfiguration('arm_namespace').perform(context).strip('/')
    if not namespace:
        raise ValueError('arm_namespace must not be empty')
    speed = float(LaunchConfiguration('max_joint_speed').perform(context))
    if not 0.01 <= speed <= 2.:
        raise ValueError('max_joint_speed must be between 0.01 and 2 rad/s')
    description = robot_description(namespace, speed,
                                    LaunchConfiguration('d435_collision').perform(context))
    package = Path(get_package_share_directory('rebotarm_ros2_control'))
    # Configure inactive so the worker can discover feedback without enabling
    # the arm. Hardware and controller activation are separate explicit steps.
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[{'robot_description': description}],
             remappings=[('/joint_states', f'/{namespace}/joint_states')]),
        Node(package='controller_manager', executable='ros2_control_node',
             namespace=namespace, output='screen', parameters=[
                 {'robot_description': description}, str(package / 'config/controllers.yaml'),
                 {'hardware_components_initial_state': {'inactive': ['RebotArmUart']}},
             ]),
        Node(package='controller_manager', executable='spawner', output='screen',
             namespace=namespace, arguments=['arm_controller', '--controller-manager',
                                            f'/{namespace}/controller_manager', '--inactive']),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('arm_namespace', default_value='rebotarm'),
        DeclareLaunchArgument('max_joint_speed', default_value='0.2'),
        DeclareLaunchArgument('d435_collision', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
