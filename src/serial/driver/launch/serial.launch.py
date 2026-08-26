import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory("rm_serial_driver")
    default_params_file = os.path.join(package_share_dir, "config", "serial_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to the serial driver parameter file",
    )

    serial_node = Node(
        package="rm_serial_driver",
        executable="node",
        name="serial_driver",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_file_arg,
        serial_node,
    ])
