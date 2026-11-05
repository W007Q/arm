"""Bridge simulation joint targets to the STM32 over UART.

Runs ``posvel_to_mit`` (converts the simulation's per-joint ``cmd/pos_vel`` into
the MIT batch the UART protocol expects) and, when a serial ``port`` is given,
the ``uart_bridge`` that talks to the STM32.

Typical use:
    # simulation side (fake driver + mujoco) already running
    ros2 launch rebotarm_uart uart_sim_bridge.launch.py port:=/dev/serial/by-id/...
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=PathJoinSubstitution([FindPackageShare("rebotarm_uart"), "config", "uart.yaml"])),
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
            DeclareLaunchArgument("port", default_value=""),
            DeclareLaunchArgument("baudrate", default_value="115200"),
            DeclareLaunchArgument("kp", default_value="60.0"),
            DeclareLaunchArgument("kd", default_value="4.0"),
            DeclareLaunchArgument("gravity_feedforward", default_value="false"),
            DeclareLaunchArgument(
                "sdk_urdf",
                default_value=PathJoinSubstitution([
                    FindPackageShare("rebotarm_description"),
                    "urdf",
                    "robot_arm.urdf",
                ]),
                description="URDF used by Pinocchio gravity feed-forward",
            ),
            DeclareLaunchArgument("launch_uart", default_value="true"),
            Node(
                package="rebotarm_uart",
                executable="posvel_to_mit",
                name="rebotarm_posvel_to_mit",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "arm_namespace": LaunchConfiguration("arm_namespace"),
                        "kp": ParameterValue(LaunchConfiguration("kp"), value_type=float),
                        "kd": ParameterValue(LaunchConfiguration("kd"), value_type=float),
                        "gravity_feedforward": ParameterValue(
                            LaunchConfiguration("gravity_feedforward"), value_type=bool
                        ),
                        "sdk_urdf": LaunchConfiguration("sdk_urdf"),
                    }
                ],
            ),
            Node(
                package="rebotarm_uart",
                executable="uart_bridge",
                name="rebotarm_uart_bridge",
                output="screen",
                condition=IfCondition(LaunchConfiguration("launch_uart")),
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "arm_namespace": LaunchConfiguration("arm_namespace"),
                        "port": LaunchConfiguration("port"),
                        "baudrate": ParameterValue(
                            LaunchConfiguration("baudrate"), value_type=int
                        ),
                        "command_source": "batch",
                    }
                ],
            ),
        ]
    )
