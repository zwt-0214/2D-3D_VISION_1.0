from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def _load_visualization_switches(params_file):
    with open(params_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    ros_params = data.get('/**', {}).get('ros__parameters', {})
    enable_visualization = bool(ros_params.get('enable_visualization', False))
    return {
        'enable_visualization': enable_visualization,
        'launch_visualization_rviz': bool(enable_visualization and ros_params.get('launch_visualization_rviz', False)),
        'launch_robot_state_publisher': bool(enable_visualization and ros_params.get('launch_robot_state_publisher', False)),
        'visualization_joint_states_topic': ros_params.get('visualization_joint_states_topic', '/joint_states'),
        'urdf_path': ros_params.get('urdf_path', ''),
        'srdf_path': ros_params.get('srdf_path', ''),
    }


def _build_nodes(context, pkg_share):
    params_file = LaunchConfiguration('params_file').perform(context)
    switches = _load_visualization_switches(params_file)

    urdf_content = ''
    srdf_content = ''
    if switches['urdf_path'] and os.path.exists(switches['urdf_path']):
        with open(switches['urdf_path'], 'r', encoding='utf-8') as f:
            urdf_content = f.read()
    if switches['srdf_path'] and os.path.exists(switches['srdf_path']):
        with open(switches['srdf_path'], 'r', encoding='utf-8') as f:
            srdf_content = f.read()

    actions = []

    planning_node = Node(
        package='lite_motion_planner',
        executable='planning_node',
        name='planning_node',
        output='screen',
        parameters=[params_file],
    )
    actions.append(planning_node)

    if switches['launch_robot_state_publisher']:
        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            remappings=[
                ('/joint_states', switches['visualization_joint_states_topic']),
            ],
            parameters=[
                {
                    'robot_description': urdf_content,
                    'publish_frequency': 100.0,
                    'ignore_timestamp': True,
                }
            ],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('use_robot_state_publisher'), "'.lower() == 'true'"])
            ),
        ))

    if switches['launch_visualization_rviz']:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[
                {'robot_description': urdf_content},
                {'robot_description_semantic': srdf_content},
            ],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ))

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('lite_motion_planner')
    default_params = os.path.join(pkg_share, 'config', 'dog2b_planner.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
    )
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    use_rsp_arg = DeclareLaunchArgument('use_robot_state_publisher', default_value='true')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'config', 'lite_motion_debug.rviz'))

    return LaunchDescription([
        params_file_arg,
        use_rviz_arg,
        use_rsp_arg,
        rviz_config_arg,
        OpaqueFunction(function=lambda context: _build_nodes(context, pkg_share)),
    ])
