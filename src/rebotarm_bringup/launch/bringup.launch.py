from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _resolve_effective_model(context, *args, **kwargs):
    model = LaunchConfiguration("model").perform(context).strip()
    if model:
        effective = model
    else:
        config_path_str = LaunchConfiguration("hardware_config").perform(context)
        config_path = Path(config_path_str).expanduser() if config_path_str else None
        effective = "hybrid"
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                ros_config = yaml.safe_load(f) or {}
            effective = str(ros_config.get("default_model") or "hybrid")
    return [SetLaunchConfiguration("effective_model", effective.strip().lower())]


def generate_launch_description():
    bringup_share = FindPackageShare("rebotarm_bringup")
    hardware_config = LaunchConfiguration("hardware_config")
    model = LaunchConfiguration("model")
    channel = LaunchConfiguration("channel")
    joint_state_rate = LaunchConfiguration("joint_state_rate")
    cmd_arbitration = LaunchConfiguration("cmd_arbitration")
    arm_namespace = LaunchConfiguration("arm_namespace")
    use_rviz = LaunchConfiguration("use_rviz")
    frame_id = LaunchConfiguration("frame_id")
    ee_frame_id = LaunchConfiguration("ee_frame_id")
    disable_after_safe_home = LaunchConfiguration("disable_after_safe_home")
    effective_model = LaunchConfiguration("effective_model")

    urdf_file = PathJoinSubstitution([
        FindPackageShare("rebotarm_description"), "urdf", "rebotarm_with_d435.urdf.xacro"
    ])
    rviz_config = PathJoinSubstitution([bringup_share, "rviz", "rebotarm.rviz"])
    robot_description = ParameterValue(Command(["xacro ", urdf_file, " d435_internal_tf:=false"]), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hardware_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("rebotarm_description"), "config", "rebotarm_hardware.yaml"]
                ),
            ),
            DeclareLaunchArgument("auto_enable", default_value="true"),
            DeclareLaunchArgument("safe_home_on_shutdown", default_value="true"),
            DeclareLaunchArgument("provide_arm_services", default_value="true"),
            DeclareLaunchArgument("model", default_value=""),
            DeclareLaunchArgument("channel", default_value=""),
            DeclareLaunchArgument("joint_state_rate", default_value="100.0"),
            DeclareLaunchArgument("cmd_arbitration", default_value="reject"),
            DeclareLaunchArgument("arm_command_source", default_value="batch"),
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("frame_id", default_value="base_link"),
            DeclareLaunchArgument("ee_frame_id", default_value="gripper_tcp"),
            DeclareLaunchArgument("disable_after_safe_home", default_value="true"),
            OpaqueFunction(function=_resolve_effective_model),
            Node(
                package="rebotarmcontroller",
                executable="reBotArmController",
                name="reBotArmController",
                output="screen",
                on_exit=Shutdown(reason="reBotArmController exited"),
                parameters=[
                    {
                        "auto_enable": ParameterValue(LaunchConfiguration("auto_enable"), value_type=bool),
                        "safe_home_on_shutdown": ParameterValue(LaunchConfiguration("safe_home_on_shutdown"), value_type=bool),
                        "provide_arm_services": ParameterValue(LaunchConfiguration("provide_arm_services"), value_type=bool),
                        "hardware_config": hardware_config,
                        "model": model,
                        "channel": channel,
                        "joint_state_rate": joint_state_rate,
                        "cmd_arbitration": cmd_arbitration,
                        "arm_command_source": LaunchConfiguration("arm_command_source"),
                        "arm_namespace": arm_namespace,
                        "frame_id": frame_id,
                        "ee_frame_id": ee_frame_id,
                        "disable_after_safe_home": ParameterValue(
                            disable_after_safe_home,
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/joint_states", ["/", arm_namespace, "/joint_states"])],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
