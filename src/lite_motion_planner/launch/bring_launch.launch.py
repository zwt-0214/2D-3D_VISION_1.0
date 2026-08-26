import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _pkg_launch(package_name, launch_name):
    return PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package_name), 'launch', launch_name)
    )


def generate_launch_description():
    pose_estimator_share = get_package_share_directory('pose_estimator')
    lite_motion_share = get_package_share_directory('lite_motion_planner')

    pose_params_file = os.path.join(pose_estimator_share, 'config', 'params.yaml')
    planner_params_file = os.path.join(lite_motion_share, 'config', 'engineer_0520_planner.yaml')
    rviz_config_file = os.path.join(lite_motion_share, 'config', 'lite_motion_debug_safe.rviz')

    nuc_arg = DeclareLaunchArgument(
        'NUC',
        default_value='false',
        description='false=Jiaolong laptop profile, true=NUC profile',
    )
    pose_params_file_arg = DeclareLaunchArgument(
        'pose_params_file',
        default_value=pose_params_file,
        description='Parameters file for pose_estimator',
    )
    planner_params_file_arg = DeclareLaunchArgument(
        'planner_params_file',
        default_value=planner_params_file,
        description='Parameters file for lite_motion_planner/tzb_catch',
    )
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    use_robot_state_publisher_arg = DeclareLaunchArgument(
        'use_robot_state_publisher',
        default_value='true',
    )
    rviz_config_arg = DeclareLaunchArgument('rviz_config', default_value=rviz_config_file)

    orbbec_camera_launch = IncludeLaunchDescription(
        _pkg_launch('orbbec_camera', 'gemini_330_series.launch.py'),
        launch_arguments={
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_left_ir': 'false',
            'enable_right_ir': 'false',
            'enable_accel': 'false',
            'enable_gyro': 'false',
            'depth_registration': 'true',
            'align_target_stream': 'COLOR',
            'enable_point_cloud': 'true',
            'enable_colored_point_cloud': 'true',
            'ordered_pc': 'true',
            'color_width': '0',
            'color_height': '0',
            'color_fps': '0',
            'depth_width': '0',
            'depth_height': '0',
            'depth_fps': '0',
        }.items(),
    )

    yolo_launch = IncludeLaunchDescription(
        _pkg_launch('yolo_node', 'yolo_node.launch.py'),
        launch_arguments={
            'NUC': LaunchConfiguration('NUC'),
        }.items(),
    )

    pose_estimator_launch = IncludeLaunchDescription(
        _pkg_launch('pose_estimator', 'pose_estimator.launch.py'),
        launch_arguments={
            'params_file': LaunchConfiguration('pose_params_file'),
            'launch_camera': 'false',
            'NUC': LaunchConfiguration('NUC'),
        }.items(),
    )

    tzb_catch_launch = IncludeLaunchDescription(
        _pkg_launch('lite_motion_planner', 'tzb_catch.launch.py'),
        launch_arguments={
            'params_file': LaunchConfiguration('planner_params_file'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_robot_state_publisher': LaunchConfiguration('use_robot_state_publisher'),
            'rviz_config': LaunchConfiguration('rviz_config'),
        }.items(),
    )

    return LaunchDescription([
        nuc_arg,
        pose_params_file_arg,
        planner_params_file_arg,
        use_rviz_arg,
        use_robot_state_publisher_arg,
        rviz_config_arg,
        orbbec_camera_launch,
        TimerAction(period=1.0, actions=[yolo_launch]),
        TimerAction(period=2.0, actions=[pose_estimator_launch]),
        TimerAction(period=3.0, actions=[tzb_catch_launch]),
    ])
