import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    LogInfo, ExecuteProcess, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('pose_estimator')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    return LaunchDescription([

        # =================== 启动参数 ===================
        DeclareLaunchArgument('params_file', default_value=params_file),
        DeclareLaunchArgument('launch_camera', default_value='false',
                              description='是否同时启动 Orbbec Gemini 330 相机'),

        # =================== Orbbec Gemini 330 ===================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('orbbec_camera'),
                '/launch/gemini_330_series.launch.py'
            ]),
            launch_arguments={
                'camera_name':                  'camera',
                'enable_color':                 'true',
                'enable_depth':                 'true',
                'enable_left_ir':               'false',
                'enable_right_ir':              'false',
                'enable_accel':                 'false',
                'enable_gyro':                  'false',

                # ---- 点云与对齐: 输出 /camera/depth_registered/points ----
                'depth_registration':           'true',
                'align_target_stream':          'COLOR',
                'enable_point_cloud':           'true',
                'enable_colored_point_cloud':   'true',
                'ordered_pc':                   'true',

                # ---- 分辨率: 0=使用设备默认 profile；如需固定可改为 640/480/30 ----
                'color_width':                  '0',
                'color_height':                 '0',
                'color_fps':                    '0',
                'depth_width':                  '0',
                'depth_height':                 '0',
                'depth_fps':                    '0',
            }.items(),
            condition=IfCondition(LaunchConfiguration('launch_camera')),
        ),

        # =================== RealSense D435i (旧配置, 已停用) ===================
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         get_package_share_directory('realsense2_camera'),
        #         '/launch/rs_launch.py'
        #     ]),
        #     launch_arguments={
        #         'pointcloud.enable':          'true',
        #         'align_depth.enable':         'true',
        #         'pointcloud.ordered_pc':      'true',
        #         'depth_module.depth_profile': '640,480,30',
        #         'rgb_camera.color_profile':   '640,480,30',
        #         'enable_infra1':              'false',
        #         'enable_infra2':              'false',
        #         'enable_gyro':                'false',
        #         'enable_accel':               'false',
        #     }.items(),
        #     condition=IfCondition(LaunchConfiguration('launch_camera')),
        # ),

        # =================== 延迟诊断: 3秒后打印话题列表 ===================
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=['bash', '-c',
                         'echo "\\n===== 当前点云相关话题 =====" && '
                         'ros2 topic list 2>/dev/null | grep -E "points|camera_info|depth" && '
                         'echo "=========================\\n"'],
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('launch_camera')),
                ),
            ],
        ),

        # =================== 位姿估计节点 ===================
        Node(
            package='pose_estimator',
            executable='pose_estimation_node',
            name='pose_estimation_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),

        LogInfo(msg=[
            '\n============================================\n',
            '  pose_estimator 已启动\n',
            '  如果单独启动相机, 务必使用:\n',
            '    ros2 launch orbbec_camera gemini_330_series.launch.py \\\n',
            '      depth_registration:=true \\\n',
            '      align_target_stream:=COLOR \\\n',
            '      enable_point_cloud:=true \\\n',
            '      enable_colored_point_cloud:=true \\\n',
            '      ordered_pc:=true\n',
            '============================================\n',
        ]),
    ])
