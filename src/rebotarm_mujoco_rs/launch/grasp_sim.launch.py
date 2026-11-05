from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("rebotarm_mujoco_rs"), "launch", "mujoco_rs.launch.py"
            ])),
            launch_arguments={
                "arm_namespace": LaunchConfiguration("arm_namespace"),
                "model_path": PathJoinSubstitution([
                    FindPackageShare("rebotarm_description"), "mjcf", "grasp_scene.xml"
                ]),
                "authority_mode": "true",
                "simulation_mode": "physics",
                "subscribe_commands": "true",
                "provide_gripper_services": "true",
                "gripper_kp": "120.0",
                "gripper_kd": "8.0",
                "gripper_force_limit": "2.0",
                "use_viewer": LaunchConfiguration("use_viewer"),
                "enable_task_tools": "true",
            }.items(),
        ),
    ])
