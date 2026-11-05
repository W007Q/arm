from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration("arm_namespace")
    return LaunchDescription([
        DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("detector_backend", default_value="yolo"),
        DeclareLaunchArgument("detection_source", default_value="camera_rgbd"),
        DeclareLaunchArgument("yolo_model", default_value="models/best.pt"),
        DeclareLaunchArgument("yolo_confidence", default_value="0.45"),
        DeclareLaunchArgument("yolo_iou", default_value="0.50"),
        DeclareLaunchArgument("yolo_imgsz", default_value="640"),
        DeclareLaunchArgument("yolo_device", default_value="0"),
        DeclareLaunchArgument("yolo_class_map", default_value="{}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"
            ])),
            launch_arguments={
                "camera_namespace": LaunchConfiguration("camera_namespace"),
                "camera_name": LaunchConfiguration("camera_name"),
                "base_frame_id": "sensor_frame",
                "enable_color": "true",
                "enable_depth": "true",
                "enable_sync": "true",
                "enable_rgbd": "true",
                "align_depth.enable": "true",
            }.items(),
        ),
        Node(
            package="rebotarm_vision",
            executable="rgbd_detector",
            name="rebotarm_rgbd_detector",
            output="screen",
            parameters=[
                PathJoinSubstitution([FindPackageShare("rebotarm_vision"), "config", "d435.yaml"]),
                {
                    "arm_namespace": namespace,
                    "base_frame": LaunchConfiguration("base_frame"),
                    "detector_backend": LaunchConfiguration("detector_backend"),
                    "detection_source": LaunchConfiguration("detection_source"),
                    "yolo_model": LaunchConfiguration("yolo_model"),
                    "yolo_confidence": LaunchConfiguration("yolo_confidence"),
                    "yolo_iou": LaunchConfiguration("yolo_iou"),
                    "yolo_imgsz": ParameterValue(LaunchConfiguration("yolo_imgsz"), value_type=int),
                    # 强制为字符串，避免 GPU 编号 0 被 ROS 推断成整数。
                    "yolo_device": ParameterValue(LaunchConfiguration("yolo_device"), value_type=str),
                    # JSON 作为字符串交给检测节点解析，避免 launch 当作 YAML 字典。
                    "yolo_class_map": ParameterValue(LaunchConfiguration("yolo_class_map"), value_type=str),
                },
            ],
        ),
    ])
