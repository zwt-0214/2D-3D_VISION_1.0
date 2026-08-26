#!/usr/bin/env python3
"""
一体化 bringup: Orbbec Gemini 330 相机 + YOLO 检测 + pose_estimator 位姿估计。

用法:
  ros2 launch pose_estimator perception_bringup.launch.py

可选参数:
  launch_camera:=false           # 相机已在别处单独启动
  camera_name:=camera            # 相机命名空间 (改动时需同步 params.yaml 的话题前缀)
  params_file:=/path/to/params.yaml

注意:
  - pose_estimator 内部也有 launch_camera 开关，这里统一由本文件启动相机，
    因此强制传 launch_camera:=false 给 pose_estimator，避免重复启动。
  - 相机参数与 pose_estimator 预期一致: depth_registration + 彩色对齐 + 彩色有序点云。
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pose_estimator_pkg = get_package_share_directory('pose_estimator')
    orbbec_camera_pkg = get_package_share_directory('orbbec_camera')
    yolo_node_pkg = get_package_share_directory('yolo_node')

    default_params = os.path.join(pose_estimator_pkg, 'config', 'params.yaml')

    camera_name = LaunchConfiguration('camera_name')
    launch_camera = LaunchConfiguration('launch_camera')
    params_file = LaunchConfiguration('params_file')

    declared_args = [
        DeclareLaunchArgument('camera_name', default_value='camera',
                              description='相机命名空间'),
        DeclareLaunchArgument('launch_camera', default_value='true',
                              description='是否启动 Orbbec Gemini 330 相机'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='pose_estimator 参数文件路径'),
    ]

    # ---- 1. Orbbec Gemini 330 相机 (深度对齐到彩色 + 彩色有序点云) ----
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(orbbec_camera_pkg, 'launch', 'gemini_330_series.launch.py')
        ]),
        launch_arguments={
            'camera_name': camera_name,
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_left_ir': 'false',
            'enable_right_ir': 'false',
            'enable_accel': 'false',
            'enable_gyro': 'false',
            # ---- 点云与对齐: 输出彩色对齐的注册点云 ----
            'depth_registration': 'true',
            'align_target_stream': 'COLOR',
            'enable_point_cloud': 'true',
            'enable_colored_point_cloud': 'true',
            'ordered_pc': 'true',
        }.items(),
        condition=IfCondition(launch_camera),
    )

    # ---- 2. YOLO 检测节点 ----
    yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(yolo_node_pkg, 'launch', 'yolo_node.launch.py')
        ]),
        launch_arguments={
            'image_topic': ['/', camera_name, '/color/image_raw'],
            'detection_topic': '/yolo/detections',
        }.items(),
    )

    # ---- 3. pose_estimator (launch_camera=false 避免重复启动相机) ----
    pose = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pose_estimator_pkg, 'launch', 'pose_estimator.launch.py')
        ]),
        launch_arguments={
            'launch_camera': 'false',
            'params_file': params_file,
        }.items(),
    )

    return LaunchDescription(declared_args + [
        camera,
        yolo,
        pose,
        LogInfo(msg='perception bringup 已启动: 相机 + YOLO + pose_estimator'),
    ])
