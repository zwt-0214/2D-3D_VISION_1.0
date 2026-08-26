from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('lite_motion_planner')
    default_params = os.path.join(pkg_share, 'config', 'engineer_0520_planner.yaml')

    params_file_arg = DeclareLaunchArgument('params_file', default_value=default_params)
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    use_rsp_arg = DeclareLaunchArgument('use_robot_state_publisher', default_value='true')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'config', 'lite_motion_debug_safe.rviz'))

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'planning_node.launch.py')),
        launch_arguments={
            'params_file': LaunchConfiguration('params_file'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_robot_state_publisher': LaunchConfiguration('use_robot_state_publisher'),
            'rviz_config': LaunchConfiguration('rviz_config'),
        }.items(),
    )

    tzb_catch_node = Node(
        package='lite_motion_planner',
        executable='tzb_catch',
        name='tzb_catch',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        use_rviz_arg,
        use_rsp_arg,
        rviz_config_arg,
        planner_launch,
        tzb_catch_node,
    ])
