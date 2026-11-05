from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", description="STM32 UART device"),
        DeclareLaunchArgument("config", default_value=PathJoinSubstitution([
            FindPackageShare("rebotarm_uart"), "config", "uart_grasp.yaml"
        ])),
        DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
        # 启动零点和 URDF 零点未标定前必须保持关闭；确认标定后显式传 true。
        DeclareLaunchArgument("gravity_feedforward", default_value="false"),
        DeclareLaunchArgument(
            "sdk_urdf",
            default_value=PathJoinSubstitution([
                FindPackageShare("rebotarm_description"), "urdf", "robot_arm.urdf"
            ]),
            description="URDF used by Pinocchio gravity feed-forward",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("rebotarm_uart"), "launch", "uart_sim_bridge.launch.py"
            ])),
            launch_arguments={
                "port": LaunchConfiguration("port"),
                "config": LaunchConfiguration("config"),
                "arm_namespace": LaunchConfiguration("arm_namespace"),
                "gravity_feedforward": LaunchConfiguration("gravity_feedforward"),
                "sdk_urdf": LaunchConfiguration("sdk_urdf"),
                "launch_uart": "true",
            }.items(),
        ),
    ])
