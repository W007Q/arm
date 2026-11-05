from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("arm_namespace")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    motion_mode = LaunchConfiguration("motion_mode")
    transport = LaunchConfiguration("transport")
    python_executable = LaunchConfiguration("python_executable")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=""),
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("port", default_value="8081"),
            DeclareLaunchArgument("motion_mode", default_value="locked"),
            DeclareLaunchArgument("transport", default_value="streamable-http"),
            DeclareLaunchArgument(
                "python_executable",
                default_value="/usr/bin/python3",
                description=(
                    "Python interpreter used to run the MCP server. The default "
                    "uses system Python; override for an environment containing fastmcp."
                ),
            ),
            Node(
                package="rebotarm_agent",
                executable="rebotarm_mcp_server",
                name="rebotarm_mcp_server",
                output="screen",
                prefix=[python_executable],
                arguments=[
                    "--config", LaunchConfiguration("config"),
                    "--namespace",
                    namespace,
                    "--host",
                    host,
                    "--port",
                    port,
                    "--motion-mode",
                    motion_mode,
                    "--transport",
                    transport,
                ],
            ),
        ]
    )
