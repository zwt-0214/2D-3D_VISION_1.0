#!/usr/bin/env python3
"""
Launch file for YOLO Detector Node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 声明启动参数
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/zwt/vision/ultralytics/results/yolo11n2/weights/openvino',
        description='Path to OpenVINO model directory'
    )

    yolo_python_arg = DeclareLaunchArgument(
        'yolo_python',
        default_value='/home/zwt/miniconda3/envs/foundationpose/bin/python',
        description='Python executable for YOLO node runtime'
    )

    ultralytics_root_arg = DeclareLaunchArgument(
        'ultralytics_root',
        default_value='/home/zwt/vision/src/ultralytics',
        description='Local ultralytics repository root path'
    )
    
    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.78',
        description='Confidence threshold for detections'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.45',
        description='IOU threshold for NMS'
    )

    publish_conf_threshold_arg = DeclareLaunchArgument(
        'publish_conf_threshold',
        default_value='0.78',
        description='Only publish when max confidence is above this threshold'
    )

    publish_hold_seconds_arg = DeclareLaunchArgument(
        'publish_hold_seconds',
        default_value='0.0',
        description='Required hold time above publish_conf_threshold before publishing'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device for inference (cpu or gpu)'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        # default_value='camera/camera/color/image_raw',

        description='Input RGB image topic from RealSense camera'
    )
    
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='yolo/detections',
        description='Output detection results topic'
    )
    
    annotated_image_topic_arg = DeclareLaunchArgument(
        'annotated_image_topic',
        default_value='yolo/annotated_image',
        description='Output annotated image topic'
    )

    left_or_right_arg = DeclareLaunchArgument(
        'left_or_right',
        default_value='0',
        description='For the two largest haomu boxes: 0 selects left box, 1 selects right box'
    )

    timing_log_enable_arg = DeclareLaunchArgument(
        'timing_log_enable',
        default_value='true',
        description='Print YOLO stage timing to terminal'
    )

    timing_log_throttle_sec_arg = DeclareLaunchArgument(
        'timing_log_throttle_sec',
        default_value='0.5',
        description='Throttle interval for YOLO timing logs'
    )

    image_queue_depth_arg = DeclareLaunchArgument(
        'image_queue_depth',
        default_value='2',
        description='Input image subscription queue depth; 2 keeps only latest frames while preventing backlog'
    )

    nuc_arg = DeclareLaunchArgument(
        'NUC',
        default_value='false',
        description='false=机械革命蛟龙16Q 8C/16T, true=NUC12 i7 12C/20T'
    )

    bbox_area_history_len_arg = DeclareLaunchArgument(
        'bbox_area_history_len',
        default_value='10',
        description='History length for haomu bbox area smoothing'
    )

    bbox_area_smoothing_enable_arg = DeclareLaunchArgument(
        'bbox_area_smoothing_enable',
        default_value='true',
        description='Use averaged bbox area for haomu selection to suppress one-frame jumps'
    )
    
    # YOLO 检测节点
    yolo_node = Node(
        package='yolo_node',
        executable='yolo_detector.py',
        prefix=[LaunchConfiguration('yolo_python')],
        name='yolo_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'publish_conf_threshold': LaunchConfiguration('publish_conf_threshold'),
            'publish_hold_seconds': LaunchConfiguration('publish_hold_seconds'),
            'device': LaunchConfiguration('device'),
            'image_topic': LaunchConfiguration('image_topic'),
            'detection_topic': LaunchConfiguration('detection_topic'),
            'annotated_image_topic': LaunchConfiguration('annotated_image_topic'),
            'left_or_right': LaunchConfiguration('left_or_right'),
            'timing_log_enable': LaunchConfiguration('timing_log_enable'),
            'timing_log_throttle_sec': LaunchConfiguration('timing_log_throttle_sec'),
            'image_queue_depth': LaunchConfiguration('image_queue_depth'),
            'NUC': LaunchConfiguration('NUC'),
            'bbox_area_history_len': LaunchConfiguration('bbox_area_history_len'),
            'bbox_area_smoothing_enable': LaunchConfiguration('bbox_area_smoothing_enable'),
        }]
    )
    
    return LaunchDescription([
        model_path_arg,
        yolo_python_arg,
        ultralytics_root_arg,
        conf_threshold_arg,
        iou_threshold_arg,
        publish_conf_threshold_arg,
        publish_hold_seconds_arg,
        device_arg,
        image_topic_arg,
        detection_topic_arg,
        annotated_image_topic_arg,
        left_or_right_arg,
        timing_log_enable_arg,
        timing_log_throttle_sec_arg,
        image_queue_depth_arg,
        nuc_arg,
        bbox_area_history_len_arg,
        bbox_area_smoothing_enable_arg,
        SetEnvironmentVariable(
            name='YOLO_NODE_PYTHON',
            value=LaunchConfiguration('yolo_python')
        ),
        SetEnvironmentVariable(
            name='ULTRALYTICS_ROOT',
            value=LaunchConfiguration('ultralytics_root')
        ),
        SetEnvironmentVariable(
            name='PYTHONNOUSERSITE',
            value='1'
        ),
        yolo_node,
    ])
