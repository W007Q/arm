import os
from pathlib import Path
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from importlib.machinery import SourceFileLoader
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

launch_common = SourceFileLoader(
    "moveit_launch_common",
    os.path.join(os.path.dirname(__file__), "moveit_launch_common.py"),
).load_module()
moveit_parameters = launch_common.moveit_parameters


def _default_model():
    try:
        path = Path(
            get_package_share_directory("rebotarm_description")
        ) / "config" / "rebotarm_hardware.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return str((yaml.safe_load(f) or {}).get("default_model") or "hybrid")
    except (PackageNotFoundError, OSError):
        pass
    return "hybrid"


def generate_launch_description():
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz configuration file in rebotarm_moveit_config/launch",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Start RViz with the MoveIt motion planning plugin",
    )
    arm_namespace_arg = DeclareLaunchArgument(
        "arm_namespace",
        default_value="rebotarm",
        description="Namespace used by an already-running reBotArmController",
    )
    model_arg = DeclareLaunchArgument(
        "model",
        default_value=_default_model(),
        description="Robot model to load: hybrid",
    )
    d435_collision_arg = DeclareLaunchArgument(
        "d435_collision",
        default_value="true",
        description="Include the calibrated D435 collision proxy in MoveIt",
    )

    return LaunchDescription(
        [
            rviz_config_arg,
            use_rviz_arg,
            arm_namespace_arg,
            model_arg,
            d435_collision_arg,
            DeclareLaunchArgument(
                "trajectory_controller", default_value="",
                description="Empty for task_server, arm_controller for the opt-in ros2_control path",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    model = LaunchConfiguration("model").perform(context).strip().lower()
    if model != "hybrid":
        raise ValueError(f"Only the hybrid DM/PA43 model is supported: {model!r}")
    arm_namespace = LaunchConfiguration("arm_namespace")
    namespace_value = arm_namespace.perform(context).strip('/')
    d435_collision = LaunchConfiguration("d435_collision").perform(context).strip().lower()

    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_moveit_config")
        .robot_description(
            file_path="config/rebotarm_hybrid.urdf.xacro",
            # 新 CAD 已包含实体 D435；碰撞开关与执行端保持一致。
            mappings={"d435_internal_tf": "false", "d435_collision": d435_collision},
        )
        .robot_description_semantic(file_path="config/rebotarm_hybrid.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_hardware_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    moveit_params = launch_common.hardware_controller_namespace(
        moveit_parameters(moveit_config), namespace_value,
        LaunchConfiguration("trajectory_controller").perform(context))
    # 真机不能跳到随机扰动后的起点。Humble 去掉自动挪起点的适配器，
    # 由 OMPL 自身的起点碰撞检查直接拒绝；其他版本的 Check 适配器只检查。
    if os.environ.get('ROS_DISTRO') == 'humble':
        adapters = moveit_params['ompl']['request_adapters'].split()
        moveit_params['ompl']['request_adapters'] = ' '.join(
            name for name in adapters if not name.endswith('/FixStartStateCollision'))

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
        remappings=[("/joint_states", ["/", arm_namespace, "/joint_states"])],
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("rebotarm_moveit_config"),
            "launch",
            LaunchConfiguration("rviz_config"),
        ]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        parameters=[moveit_params],
        remappings=[("/joint_states", ["/", arm_namespace, "/joint_states"])],
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
    )

    # 真机后端已发布机械臂 TF；RealSense 发布内部光学 TF。
    # 此处只接入规划与 RViz，避免第二套 TF 覆盖真实相机外参。
    return [
        static_tf_node,
        move_group_node,
        rviz_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=move_group_node,
                on_exit=[
                    EmitEvent(event=Shutdown(reason="move_group exited"))
                ],
            )
        ),
    ]
