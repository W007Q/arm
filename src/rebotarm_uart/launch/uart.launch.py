from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port',description='STM32 UART device path'),
        DeclareLaunchArgument('config',default_value=PathJoinSubstitution([FindPackageShare('rebotarm_uart'),'config','uart.yaml'])),
        DeclareLaunchArgument('baudrate',default_value='115200'),
        DeclareLaunchArgument('arm_namespace',default_value='rebotarm'),
        DeclareLaunchArgument('command_source',default_value='batch'),
        Node(package='rebotarm_uart',executable='uart_bridge',name='rebotarm_uart_bridge',output='screen',
             parameters=[LaunchConfiguration('config'),
                         {'port':ParameterValue(LaunchConfiguration('port'),value_type=str),
                          'baudrate':ParameterValue(LaunchConfiguration('baudrate'),value_type=int),
                          'arm_namespace':ParameterValue(LaunchConfiguration('arm_namespace'),value_type=str),
                          'command_source':ParameterValue(LaunchConfiguration('command_source'),value_type=str)}])
    ])
