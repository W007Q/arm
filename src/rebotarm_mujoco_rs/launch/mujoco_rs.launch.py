import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_list(value: str):
    parsed = yaml.safe_load(value)
    return list(parsed) if parsed is not None else []


def _launch_setup(context, *args, **kwargs):
    def get(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    namespace = get("arm_namespace")
    model_path = get("model_path")

    def get_bool(name: str) -> bool:
        return str(get(name)).strip().lower() in ("1", "true", "yes")

    enable_task_tools = get_bool("enable_task_tools")

    mujoco_node = Node(
        package="rebotarm_mujoco_rs",
        executable="mujoco_rs_sync",
        name="rebotarm_rs_mujoco",
        output="screen",
        parameters=[
            {
                "arm_namespace": namespace,
                "input_topic": get("input_topic"),
                "output_topic": get("output_topic"),
                "model_path": model_path,
                "simulation_mode": get("simulation_mode"),
                "update_rate": float(get("update_rate")),
                "smoothing_alpha": float(get("smoothing_alpha")),
                "stale_timeout": float(get("stale_timeout")),
                "arm_kp": _parse_list(get("arm_kp")),
                "arm_kd": _parse_list(get("arm_kd")),
                "arm_tau_limit": _parse_list(get("arm_tau_limit")),
                "gravity_bias_scale": float(get("gravity_bias_scale")),
                "arm_target_speed": _parse_list(get("arm_target_speed")),
                "motor_specs": get("motor_specs"),
                "motor_model_enabled": get_bool("motor_model_enabled"),
                "compliance_enabled": get_bool("compliance_enabled"),
                "compliance_scale": float(get("compliance_scale")),
                "contact_torque_threshold": float(get("contact_torque_threshold")),
                "authority_mode": get_bool("authority_mode"),
                "joint_states_topic": get("joint_states_topic"),
                "subscribe_commands": get_bool("subscribe_commands"),
                "legacy_joint_command_topics": get_bool("legacy_joint_command_topics"),
                "use_viewer": get_bool("use_viewer"),
                "gripper_kp": float(get("gripper_kp")),
                "gripper_kd": float(get("gripper_kd")),
                "gripper_tau_limit": float(get("gripper_tau_limit")),
                "gripper_force_limit": float(get("gripper_force_limit")),
                "gripper_target_speed": float(get("gripper_target_speed")),
                "provide_gripper_services": get_bool("provide_gripper_services"),
                "control_mode": get("control_mode"),
                "cartesian_target_topic": get("cartesian_target_topic"),
                "cartesian_kp": float(get("cartesian_kp")),
                "cartesian_kd": float(get("cartesian_kd")),
                "cartesian_rot_kp": float(get("cartesian_rot_kp")),
                "cartesian_rot_kd": float(get("cartesian_rot_kd")),
            }
        ],
    )

    nodes = [mujoco_node]
    if enable_task_tools:
        nodes.extend(
            [
                Node(
                    package="rebotarm_mujoco_rs",
                    executable="rs_task_server",
                    name="rebotarm_rs_task_server",
                    output="screen",
                    parameters=[
                        {"arm_namespace": namespace, "model_path": model_path}
                    ],
                ),
                Node(
                    package="rebotarm_mujoco_rs",
                    executable="rs_scene_camera",
                    name="rebotarm_rs_scene_camera",
                    output="screen",
                    parameters=[
                        {"arm_namespace": namespace, "model_path": model_path}
                    ],
                ),
                Node(
                    package="rebotarm_mujoco_rs",
                    executable="rs_scene_detector",
                    name="rebotarm_rs_scene_detector",
                    output="screen",
                    parameters=[{"arm_namespace": namespace}],
                ),
            ]
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm_rs"),
            DeclareLaunchArgument("input_topic", default_value=""),
            DeclareLaunchArgument("output_topic", default_value=""),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument("simulation_mode", default_value="physics"),
            DeclareLaunchArgument("update_rate", default_value="250.0"),
            DeclareLaunchArgument("smoothing_alpha", default_value="1.0"),
            DeclareLaunchArgument("stale_timeout", default_value="1.0"),
            DeclareLaunchArgument("use_viewer", default_value="false"),
            DeclareLaunchArgument(
                "arm_kp", default_value="[50.0, 150.0, 150.0, 50.0, 50.0, 50.0]"
            ),
            DeclareLaunchArgument(
                "arm_kd", default_value="[3.0, 10.0, 10.0, 5.0, 4.0, 4.0]"
            ),
            DeclareLaunchArgument(
                "arm_tau_limit", default_value="[36.0, 36.0, 36.0, 14.0, 14.0, 14.0]"
            ),
            DeclareLaunchArgument(
                "arm_target_speed", default_value="[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
            ),
            DeclareLaunchArgument("motor_specs", default_value=""),
            DeclareLaunchArgument("motor_model_enabled", default_value="true"),
            DeclareLaunchArgument("compliance_enabled", default_value="false"),
            DeclareLaunchArgument("compliance_scale", default_value="0.5"),
            DeclareLaunchArgument("contact_torque_threshold", default_value="1.0"),
            DeclareLaunchArgument("authority_mode", default_value="false"),
            DeclareLaunchArgument("joint_states_topic", default_value=""),
            DeclareLaunchArgument("subscribe_commands", default_value="false"),
            DeclareLaunchArgument("gravity_bias_scale", default_value="1.0"),
            DeclareLaunchArgument("legacy_joint_command_topics", default_value="false"),
            DeclareLaunchArgument("gripper_kp", default_value="300.0"),
            DeclareLaunchArgument("gripper_kd", default_value="120.0"),
            DeclareLaunchArgument("gripper_tau_limit", default_value="150.0"),
            DeclareLaunchArgument("gripper_force_limit", default_value="0.0"),
            DeclareLaunchArgument("gripper_target_speed", default_value="0.06"),#抓夾閉合速度
            DeclareLaunchArgument("provide_gripper_services", default_value="false"),
            DeclareLaunchArgument("control_mode", default_value="joint"),
            DeclareLaunchArgument("cartesian_target_topic", default_value=""),
            DeclareLaunchArgument("cartesian_kp", default_value="200.0"),
            DeclareLaunchArgument("cartesian_kd", default_value="20.0"),
            DeclareLaunchArgument("cartesian_rot_kp", default_value="20.0"),
            DeclareLaunchArgument("cartesian_rot_kd", default_value="2.0"),
            DeclareLaunchArgument("enable_task_tools", default_value="true"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
