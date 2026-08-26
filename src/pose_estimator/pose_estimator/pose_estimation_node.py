#!/usr/bin/env python3
"""Small symmetric object 6DoF pose estimation.

Pipeline: YOLO ROI -> one scene voxel grid -> one 20 mm FPFH/RANSAC
hypothesis -> trust-region Point-to-Point ICP -> previous-pose tracking.
Registration quality is measured only from scene points to the transformed
complete model surface, so invisible model faces do not dilute the score.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
import tf2_ros

import sys
import types

import numpy as np


def _prepare_open3d_import():
    """Avoid Open3D importing an incompatible Dash/Flask stack for unused visualization."""
    try:
        from dash import Dash, dcc, html  # noqa: F401
    except Exception:
        dash_stub = types.ModuleType('dash')
        dash_stub.Dash = type('Dash', (), {})
        dash_stub.dcc = types.SimpleNamespace()
        dash_stub.html = types.SimpleNamespace()
        sys.modules['dash'] = dash_stub


_prepare_open3d_import()
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
import threading
import copy
from scipy.spatial import cKDTree

# ---------- 检测消息兼容层 ----------
try:
    from yolo_node.msg import DetectionArray as DetectionMsg
    _DET_TYPE = "yolo_node"
except ImportError:
    try:
        from yolov8_msgs.msg import DetectionArray as DetectionMsg
        _DET_TYPE = "yolov8"
    except ImportError:
        try:
            from vision_msgs.msg import Detection2DArray as DetectionMsg
            _DET_TYPE = "vision"
        except ImportError:
            DetectionMsg = None
            _DET_TYPE = "none"


# ============================================================
#  工具函数
# ============================================================

def pc2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """将 PointCloud2 转为 (N, 3) float32 numpy 数组 (可写)."""
    n = msg.width * msg.height
    offsets = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

    if offsets.get('x', -1) == 0 and offsets.get('y', -1) == 4 and offsets.get('z', -1) == 8:
        xyz = raw[:, :12].copy().view(np.float32).reshape(n, 3)
    else:
        xyz = np.empty((n, 3), dtype=np.float32)
        for i, name in enumerate(('x', 'y', 'z')):
            o = offsets[name]
            xyz[:, i] = raw[:, o:o + 4].copy().view(np.float32).ravel()
    return xyz


def pc2_roi_numpy(msg: PointCloud2, r1: int, r2: int, c1: int, c2: int) -> np.ndarray:
    """
    ★ 从组织化 PointCloud2 中仅提取 ROI 区域, 返回 (M, 3) float32.
    避免解析全部 40 万点, 性能提升 10x+.
    """
    offsets = {f.name: f.offset for f in msg.fields}
    ps = msg.point_step
    # frombuffer → 只读 view; 只在 ROI 区域做 .copy()
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ps)
    roi = raw[r1:r2, c1:c2, :].copy()          # ★ 仅复制 ROI 字节
    n = roi.shape[0] * roi.shape[1]
    roi = roi.reshape(n, ps)

    if offsets.get('x', -1) == 0 and offsets.get('y', -1) == 4 and offsets.get('z', -1) == 8:
        return roi[:, :12].view(np.float32).reshape(n, 3)
    else:
        xyz = np.empty((n, 3), dtype=np.float32)
        for i, name in enumerate(('x', 'y', 'z')):
            o = offsets[name]
            xyz[:, i] = roi[:, o:o + 4].view(np.float32).ravel()
        return xyz


def extract_bbox_from_detection(det_msg, target_class: str, det_type: str):
    """从检测消息提取 bbox → (x1, y1, x2, y2) 彩色图像素坐标."""
    best = None
    best_score = 0.0

    if det_type == "yolov8":
        for det in det_msg.detections:
            if target_class and det.class_name != target_class:
                continue
            if det.score < best_score:
                continue
            best_score = det.score
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size.x
            h = det.bbox.size.y
            best = (int(cx - w / 2), int(cy - h / 2),
                    int(cx + w / 2), int(cy + h / 2))

    elif det_type == "vision":
        for det in det_msg.detections:
            score = 0.0
            cls_name = ""
            if det.results:
                r_best = max(det.results, key=lambda r: r.hypothesis.score)
                score = r_best.hypothesis.score
                cls_name = r_best.hypothesis.class_id
            if target_class and cls_name != target_class:
                continue
            if score < best_score:
                continue
            best_score = score
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            best = (int(cx - w / 2), int(cy - h / 2),
                    int(cx + w / 2), int(cy + h / 2))

    elif det_type == "yolo_node":
        for det in det_msg.detections:
            if target_class and det.class_name != target_class:
                continue
            if det.confidence < best_score:
                continue
            best_score = det.confidence
            best = (int(det.x), int(det.y),
                    int(det.x + det.width), int(det.y + det.height))

    return best, best_score


def _make_field(name, offset):
    f = PointField()
    f.name = name
    f.offset = offset
    f.datatype = PointField.FLOAT32
    f.count = 1
    return f


def stamp_to_sec(stamp) -> float:
    """ROS builtin_interfaces/Time -> seconds."""
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return 0.0


def clamp_bbox(bbox, width=None, height=None):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if width:
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width - 1), x2))
    if height:
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


# ============================================================
#  模型预处理器
# ============================================================

def _estimate_normals(cloud, radius, max_nn, camera_location=None, outward=False):
    """Estimate normals and make their direction deterministic."""
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius), max_nn=int(max_nn)))
    if camera_location is not None:
        cloud.orient_normals_towards_camera_location(np.asarray(camera_location))
    elif outward and cloud.has_normals():
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        flip = np.einsum('ij,ij->i', normals, points) < 0.0
        normals[flip] *= -1.0
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


class ModelData:
    def __init__(self, path, normal_radius, normal_max_nn, feature_radius,
                 feature_max_nn, model_auto_unit_scale=True):
        raw = o3d.io.read_point_cloud(path)
        if raw.is_empty():
            raise FileNotFoundError(f"无法读取模型: {path}")

        self.raw_count = len(raw.points)
        raw_pts0 = np.asarray(raw.points)
        self.raw_min_before_scale = raw_pts0.min(axis=0)
        self.raw_max_before_scale = raw_pts0.max(axis=0)
        self.raw_extent_before_scale = self.raw_max_before_scale - self.raw_min_before_scale
        self.model_unit_scale_applied = 1.0

        # 自动单位自检: 目标体积约 0.00117809 m^3，等效尺寸约 0.106 m。
        # 如果 PCD 最大外形尺寸明显大于 2m，通常表示以 mm 导出，自动转成 m。
        max_extent0 = float(np.max(self.raw_extent_before_scale)) if raw_pts0.size else 0.0
        if bool(model_auto_unit_scale) and max_extent0 > 2.0:
            raw.scale(0.001, center=(0.0, 0.0, 0.0))
            self.model_unit_scale_applied = 0.001

        raw_pts0 = np.asarray(raw.points)
        self.raw_min = raw_pts0.min(axis=0)
        self.raw_max = raw_pts0.max(axis=0)
        self.raw_extent = self.raw_max - self.raw_min
        self.centroid = np.asarray(raw.get_center())
        raw.translate(-self.centroid)

        # The 10k model is already at the intended density. Keep every point.
        self.cloud = raw
        _estimate_normals(
            self.cloud, normal_radius, normal_max_nn, outward=True)
        self.feature = o3d.pipelines.registration.compute_fpfh_feature(
            self.cloud,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(feature_radius), max_nn=int(feature_max_nn)))
        self.kdtree = cKDTree(np.asarray(self.cloud.points))
        print(f"[ModelData] points={len(self.cloud.points)} centroid={self.centroid}\n"
              f"  raw={self.raw_count} extent={self.raw_extent} min={self.raw_min} max={self.raw_max}\n"
              f"  unit_scale={self.model_unit_scale_applied} extent_before_scale={self.raw_extent_before_scale}\n"
              f"  FPFH: radius={float(feature_radius):.3f}m max_nn={int(feature_max_nn)}")


# ============================================================
#  主节点
# ============================================================

class PoseEstimationNode(Node):

    def __init__(self):
        super().__init__('pose_estimation_node')

        self._declare_all_params()
        p = self._p

        self.get_logger().info(f"加载模型: {p['model_path']}")
        self.model = ModelData(
            p['model_path'], p['normal_radius'], p['normal_max_nn'],
            p['feature_radius'], p['feature_max_nn'],
            p.get('model_auto_unit_scale', True))
        self.get_logger().info(
            f"模型加载完毕: raw={self.model.raw_count} used={len(self.model.cloud.points)} "
            f"extent={np.array2string(self.model.raw_extent, precision=4)} centroid={np.array2string(self.model.centroid, precision=4)} "
            f"unit_scale={self.model.model_unit_scale_applied}")

        # --------------- 状态 ---------------
        self._lock = threading.Lock()
        self._latest_det_bbox = None
        self._latest_det_raw_bbox = None
        self._latest_det_score = 0.0
        self._latest_det_stamp_sec = 0.0
        self._latest_det_recv_mono = 0.0
        self._last_timing_log_mono = 0.0
        self._last_transform = None
        self._tracking = False
        self._last_coarse_transform = None
        self._last_ransac_diag = None
        self._last_icp_diag = None
        self._frame_count = 0
        self._viz_publish_every_n = max(1, int(p['viz_publish_every_n']))
        self._viz_frame_count = 0
        self._latest_cloud_item = None
        self._cloud_item_lock = threading.Lock()
        self._cloud_process_lock = threading.Lock()

        self._pc_frame_id = None
        self._yolo_img_width = None
        self._yolo_img_height = None

        self._color_fx = None
        self._color_fy = None
        self._color_cx = None
        self._color_cy = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._first_frame_logged = False
        self._last_global_ransac_mono = 0.0
        self._last_reg_timing = {'reg_mode': 'IDLE', 'global_coarse': 0.0, 'icp_fine': 0.0}

        self._cloud_recv_count = 0
        self._det_recv_count = 0
        self._caminfo_recv_count = 0

        # --------------- QoS ---------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE)
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._cb_group_cloud = ReentrantCallbackGroup()
        self._cb_group_fast = ReentrantCallbackGroup()

        # --------------- 订阅 ---------------
        self.sub_cloud = self.create_subscription(
            PointCloud2, p['pointcloud_topic'], self._cb_cloud, sensor_qos,
            callback_group=self._cb_group_cloud)

        self.sub_cam_info = self.create_subscription(
            CameraInfo, p['camera_info_topic'], self._cb_cam_info, 10,
            callback_group=self._cb_group_fast)

        if DetectionMsg is not None:
            self.sub_det = self.create_subscription(
                DetectionMsg, p['detection_topic'], self._cb_detection, 10,
                callback_group=self._cb_group_fast)
            self.get_logger().info(f"检测消息类型: {_DET_TYPE}")
        else:
            self.get_logger().warn("未找到检测消息包, 检测订阅已禁用")

        # --------------- 发布 ---------------
        # ★ RViz 可视化话题列表 ★
        # ────────────────────────────────────────────────────────
        # 话题                              类型           用途
        # ────────────────────────────────────────────────────────
        # ~/roi_cloud                       PointCloud2    降采样后的 ROI 点云
        # ~/coarse_model_marker             Marker         进入 ICP 的粗姿态 (蓝色)
        # ~/model_marker                    Marker         配准后的模型点云 (绿色)
        # object_pose_topic                 PoseStamped    6DoF 位姿
        # ~/object_pose                     PoseStamped    6DoF 位姿（兼容旧话题）
        # /camera/depth_registered/points   PointCloud2    Orbbec 彩色对齐/注册点云
        # /camera/camera/depth/color/points PointCloud2    D435i 旧原始完整点云
        # /yolo/annotated_image             Image          YOLO 检测标注图
        # ────────────────────────────────────────────────────────
        self.pub_pose = self.create_publisher(PoseStamped, p['object_pose_topic'], 10)
        self.pub_roi = self.create_publisher(PointCloud2, '~/roi_cloud', 1)
        self.pub_marker = self.create_publisher(Marker, '~/model_marker', marker_qos)
        self.pub_coarse_marker = self.create_publisher(Marker, '~/coarse_model_marker', marker_qos)
        self.pub_coarse_pose = self.create_publisher(PoseStamped, '~/coarse_pose', 1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.srv_reset = self.create_service(
            Trigger, '~/reset_tracking', self._srv_reset,
            callback_group=self._cb_group_fast)

        self._diag_timer = self.create_timer(
            5.0, self._diagnostic_check, callback_group=self._cb_group_fast)

        self.get_logger().info(
            f"节点已启动, 等待数据... "
            f"订阅点云: {p['pointcloud_topic']} | "
            f"订阅检测: {p['detection_topic']} | "
            f"订阅CamInfo: {p['camera_info_topic']} | "
            f"发布位姿: {p['object_pose_topic']} | "
            f"可视化发布: ROI={p['publish_roi_cloud']} Marker={p['publish_model_marker']} Coarse={p['publish_coarse_marker']} (every_n={self._viz_publish_every_n})")

    # ----------------------------------------------------------------
    #  参数声明
    # ----------------------------------------------------------------
    def _declare_all_params(self):
        decl = self.declare_parameter
        decl('model_path', '/home/zwt/E_1w.pcd')
        decl('pointcloud_topic', '/camera/depth_registered/points')
        decl('detection_topic', '/yolo/detections')
        decl('target_class', '')
        decl('object_pose_topic', '/pose_estimator/object_pose')
        decl('pose_yaw', 0.0)
        decl('camera_info_topic', '/camera/color/camera_info')
        decl('camera_frame', '')
        decl('scene_voxel_size', 0.002)
        decl('normal_radius', 0.010)
        decl('normal_max_nn', 50)
        decl('feature_radius', 0.020)
        decl('feature_max_nn', 100)
        decl('model_auto_unit_scale', True)
        decl('ransac_n', 3)
        decl('ransac_max_iter', 50000)
        decl('ransac_dist_thresh', 0.008)
        decl('ransac_mutual_filter', False)
        decl('ransac_confidence', 0.999)
        decl('ransac_edge_length_ratio', 0.85)
        decl('global_ransac_enable', True)
        decl('global_ransac_retry_interval_sec', 0.05)
        decl('scene_coverage_dist_thresh', 0.004)
        decl('coarse_min_scene_coverage', 0.25)
        decl('coarse_max_scene_median', 0.012)
        decl('icp_min_scene_coverage', 0.45)
        decl('icp_max_scene_median', 0.006)
        decl('tracking_min_scene_coverage', 0.35)
        decl('tracking_max_scene_median', 0.008)
        decl('icp_dist_thresh', 0.008)
        decl('icp_max_iter', 60)
        decl('icp_relative_fitness', 5.0e-5)
        decl('icp_relative_rmse', 5.0e-5)
        decl('global_icp_max_translation', 0.025)
        decl('global_icp_max_axis_deg', 25.0)
        decl('icp_max_scene_coverage_drop', 0.02)
        decl('icp_max_scene_median_increase', 0.001)
        decl('tracking_icp_dist_thresh', 0.005)
        decl('tracking_icp_max_iter', 30)
        decl('roi_timeout_sec', 0.35)
        decl('publish_roi_cloud', True)
        decl('publish_model_marker', True)
        decl('viz_publish_every_n', 3)
        decl('publish_coarse_pose', True)
        decl('publish_coarse_marker', True)
        decl('bbox_expand_ratio', 0.10)
        decl('bbox_expand_ratio_long', 0.02)
        decl('bbox_expand_ratio_width', 0.02)
        decl('min_scene_points', 250)
        decl('rough_crop_pad_ratio', 0.35)

        # ---- 可信 ROI / bbox 防抖 ----
        decl('det_min_score', 0.87)
        decl('bbox_smoothing_enable', True)
        decl('bbox_ema_alpha', 0.65)

        # ---- ICP 追踪跳变保护 ----
        decl('max_tracking_translation_jump', 0.020)
        decl('max_tracking_rotation_jump_deg', 30.0)
        decl('axisymmetric_ignore_yaw_jump', True)

        # ---- 终端计时日志 ----
        decl('timing_log_enable', True)
        decl('timing_log_throttle_sec', 0.5)

        names = [
            'model_path', 'pointcloud_topic', 'detection_topic', 'target_class',
            'object_pose_topic', 'pose_yaw',
            'camera_info_topic', 'camera_frame',
            'scene_voxel_size', 'normal_radius', 'normal_max_nn', 'feature_radius',
            'feature_max_nn', 'model_auto_unit_scale',
            'ransac_n', 'ransac_max_iter', 'ransac_dist_thresh',
            'ransac_mutual_filter', 'ransac_confidence', 'ransac_edge_length_ratio',
            'global_ransac_enable', 'global_ransac_retry_interval_sec',
            'scene_coverage_dist_thresh', 'coarse_min_scene_coverage',
            'coarse_max_scene_median', 'icp_min_scene_coverage',
            'icp_max_scene_median', 'tracking_min_scene_coverage',
            'tracking_max_scene_median', 'icp_dist_thresh', 'icp_max_iter',
            'icp_relative_fitness', 'icp_relative_rmse',
            'global_icp_max_translation', 'global_icp_max_axis_deg',
            'icp_max_scene_coverage_drop', 'icp_max_scene_median_increase',
            'tracking_icp_dist_thresh', 'tracking_icp_max_iter', 'roi_timeout_sec',
            'publish_roi_cloud', 'publish_model_marker', 'viz_publish_every_n', 'publish_coarse_pose', 'publish_coarse_marker',
            'bbox_expand_ratio', 'bbox_expand_ratio_long', 'bbox_expand_ratio_width', 'min_scene_points', 'rough_crop_pad_ratio',
            'det_min_score', 'bbox_smoothing_enable', 'bbox_ema_alpha',
            'max_tracking_translation_jump', 'max_tracking_rotation_jump_deg', 'axisymmetric_ignore_yaw_jump',
            'timing_log_enable', 'timing_log_throttle_sec',
        ]
        self._p = {n: self.get_parameter(n).value for n in names}


    def _reset_tracking_state(self, reason: str, throttle: float = 1.0):
        """丢弃当前迭代位姿；下一次有效 ROI 重新进入全局粗配准。"""
        was_tracking = bool(self._tracking) or self._last_transform is not None
        self._tracking = False
        self._last_transform = None
        if was_tracking:
            self.get_logger().warn(reason, throttle_duration_sec=throttle)

    # ----------------------------------------------------------------
    #  诊断
    # ----------------------------------------------------------------
    def _diagnostic_check(self):
        self._diag_timer.cancel()
        p = self._p
        self.get_logger().info(
            f"\n{'='*55}\n"
            f"  ★ 启动诊断 (5秒内接收统计)\n"
            f"{'='*55}\n"
            f"  点云   [{p['pointcloud_topic']}]: "
            f"{'✓ ' + str(self._cloud_recv_count) + ' 帧' if self._cloud_recv_count > 0 else '✗ 未收到!'}\n"
            f"  检测   [{p['detection_topic']}]: "
            f"{'✓ ' + str(self._det_recv_count) + ' 帧' if self._det_recv_count > 0 else '✗ 未收到!'}\n"
            f"  CamInfo [{p['camera_info_topic']}]: "
            f"{'✓ ' + str(self._caminfo_recv_count) + ' 帧' if self._caminfo_recv_count > 0 else '✗ 未收到!'}\n"
            f"  内参:   {'✓ 已获取' if self._color_fx else '✗ 未获取'}\n"
            f"{'='*55}")

        if self._cloud_recv_count == 0:
            self._suggest_topics('sensor_msgs/msg/PointCloud2', 'pointcloud_topic')
        if self._caminfo_recv_count == 0:
            self._suggest_topics('sensor_msgs/msg/CameraInfo', 'camera_info_topic')

    def _suggest_topics(self, msg_type, param_name):
        try:
            topics = self.get_topic_names_and_types()
            matches = [name for name, types in topics if msg_type in types]
            if matches:
                self.get_logger().warn(
                    f"⚠ 可用的 {msg_type} 话题:\n" +
                    '\n'.join(f'  → {t}' for t in matches) +
                    f"\n  请在 params.yaml 中修改 {param_name}")
            else:
                self.get_logger().error(f"✗ 系统中没有 {msg_type} 话题!")
        except Exception:
            pass

    # ----------------------------------------------------------------
    #  frame_id
    # ----------------------------------------------------------------
    def _get_frame_id(self):
        if self._pc_frame_id:
            return self._pc_frame_id
        if self._p['camera_frame']:
            return self._p['camera_frame']
        return 'camera_depth_optical_frame'

    # ----------------------------------------------------------------
    #  回调: CameraInfo
    # ----------------------------------------------------------------
    def _cb_cam_info(self, msg: CameraInfo):
        self._caminfo_recv_count += 1
        K = msg.k
        self._color_fx = K[0]
        self._color_fy = K[4]
        self._color_cx = K[2]
        self._color_cy = K[5]
        self._yolo_img_width = msg.width
        self._yolo_img_height = msg.height
        if self._caminfo_recv_count == 1:
            self.get_logger().info(
                f"✓ CameraInfo: {msg.width}×{msg.height}, "
                f"fx={K[0]:.1f} fy={K[4]:.1f} cx={K[2]:.1f} cy={K[5]:.1f}, "
                f"frame={msg.header.frame_id}")

    # ----------------------------------------------------------------
    #  回调: 检测
    # ----------------------------------------------------------------
    def _cb_detection(self, msg):
        self._det_recv_count += 1
        bbox, score = extract_bbox_from_detection(
            msg, self._p['target_class'], _DET_TYPE)
        now_mono = time.monotonic()
        stamp_sec = stamp_to_sec(msg.header.stamp) if hasattr(msg, 'header') else 0.0

        if bbox is not None:
            bbox = clamp_bbox(bbox, self._yolo_img_width, self._yolo_img_height)

        with self._lock:
            self._latest_det_raw_bbox = bbox
            self._latest_det_score = float(score)
            self._latest_det_stamp_sec = stamp_sec
            self._latest_det_recv_mono = now_mono

            if bbox is None or score < self._p['det_min_score']:
                self._latest_det_bbox = None
                return

            if self._p['bbox_smoothing_enable'] and self._latest_det_bbox is not None:
                a = float(self._p['bbox_ema_alpha'])
                a = max(0.0, min(1.0, a))
                prev = np.asarray(self._latest_det_bbox, dtype=np.float64)
                cur = np.asarray(bbox, dtype=np.float64)
                smooth = a * cur + (1.0 - a) * prev
                self._latest_det_bbox = clamp_bbox(tuple(smooth), self._yolo_img_width, self._yolo_img_height)
            else:
                self._latest_det_bbox = bbox

    # ----------------------------------------------------------------
    #  回调: 点云 (仅保留最新帧, 不堆栈)
    # ----------------------------------------------------------------
    def _cb_cloud(self, msg: PointCloud2):
        self._cloud_recv_count += 1
        with self._cloud_item_lock:
            self._frame_count += 1
            frame_seq = self._frame_count
            self._latest_cloud_item = (msg, frame_seq)

        # 已有线程在处理时, 这里只更新 latest, 直接返回
        if not self._cloud_process_lock.acquire(blocking=False):
            return

        try:
            while True:
                with self._cloud_item_lock:
                    item = self._latest_cloud_item
                    self._latest_cloud_item = None
                if item is None:
                    break
                latest_msg, latest_seq = item
                self._process_latest_cloud(latest_msg, latest_seq)
        finally:
            self._cloud_process_lock.release()

    def _process_latest_cloud(self, msg: PointCloud2, frame_seq: int):
        # 不再按 process_every_n 跳帧；_cb_cloud 已保证只处理最新点云，避免队列堆积。
        if self._pc_frame_id is None and msg.header.frame_id:
            self._pc_frame_id = msg.header.frame_id
            self.get_logger().info(
                f"✓ 点云 frame='{self._pc_frame_id}', "
                f"size={msg.width}×{msg.height}, "
                f"point_step={msg.point_step}")

        if self._color_fx is None:
            self.get_logger().warn(
                "等待 CameraInfo ...", throttle_duration_sec=3.0)
            return

        now_mono = time.monotonic()
        with self._lock:
            bbox = self._latest_det_bbox
            det_score = self._latest_det_score
            det_stamp_sec = self._latest_det_stamp_sec
            det_recv_mono = self._latest_det_recv_mono

        roi_timeout_sec = float(self._p.get('roi_timeout_sec', 0.50))
        roi_stale = (det_recv_mono <= 0.0) or ((now_mono - det_recv_mono) > roi_timeout_sec)
        if bbox is None or roi_stale:
            if roi_stale:
                self._reset_tracking_state(
                    f"YOLO ROI 超时 {now_mono - det_recv_mono:.3f}s > {roi_timeout_sec:.3f}s，已丢弃追踪位姿；下次有效 ROI 重新粗定位",
                    throttle=1.0)
                with self._lock:
                    self._latest_det_bbox = None
            return

        t0 = time.monotonic()
        try:
            self._process(msg, bbox, det_score=det_score, det_stamp_sec=det_stamp_sec, det_recv_mono=det_recv_mono)
        except Exception as e:
            self.get_logger().error(f"处理异常: {e}", throttle_duration_sec=2.0)
            import traceback
            traceback.print_exc()

        elapsed = (time.monotonic() - t0) * 1000
        if not self._p['timing_log_enable']:
            mode = "TRACK" if self._tracking else "GLOBAL"
            self.get_logger().info(
                f"[{mode}] 总耗时 {elapsed:.0f} ms",
                throttle_duration_sec=0.5)

    # ----------------------------------------------------------------
    #  ★ 3D→2D 投影裁剪 (精确)
    # ----------------------------------------------------------------
    def _crop_by_projection(self, points_3d: np.ndarray, bbox) -> np.ndarray:
        fx = self._color_fx
        fy = self._color_fy
        cx = self._color_cx
        cy = self._color_cy

        x1, y1, x2, y2 = bbox
        legacy_expand = self._p.get('bbox_expand_ratio', 0.15)
        expand_long = self._p.get('bbox_expand_ratio_long', legacy_expand)
        expand_width = self._p.get('bbox_expand_ratio_width', legacy_expand)

        bw = x2 - x1
        bh = y2 - y1
        if bw >= bh:
            expand_x, expand_y = expand_long, expand_width
        else:
            expand_x, expand_y = expand_width, expand_long

        x1_e = x1 - bw * expand_x
        y1_e = y1 - bh * expand_y
        x2_e = x2 + bw * expand_x
        y2_e = y2 + bh * expand_y

        valid = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0.001)
        pts = points_3d[valid]
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        Z = pts[:, 2]
        u = fx * pts[:, 0] / Z + cx
        v = fy * pts[:, 1] / Z + cy

        mask = (u >= x1_e) & (u <= x2_e) & (v >= y1_e) & (v <= y2_e)
        return pts[mask]

    # ----------------------------------------------------------------
    #  ★ 组织化点云粗裁剪区域计算
    # ----------------------------------------------------------------
    def _compute_rough_roi(self, msg: PointCloud2, bbox):
        """
        将 YOLO bbox (彩色图坐标) 近似映射到 depth 图像坐标,
        加大量 padding, 用于从组织化点云中仅提取小区域.
        返回 (r1, r2, c1, c2) 行列范围.
        """
        pc_H, pc_W = msg.height, msg.width
        x1, y1, x2, y2 = bbox

        if self._yolo_img_width and self._yolo_img_height:
            sx = pc_W / self._yolo_img_width
            sy = pc_H / self._yolo_img_height
        else:
            sx = sy = 1.0

        # 映射到 depth 像素
        dx1 = int(x1 * sx)
        dy1 = int(y1 * sy)
        dx2 = int(x2 * sx)
        dy2 = int(y2 * sy)

        # ★ 大量 padding: depth/color 有基线偏移 + FOV 不同
        pad_ratio = self._p['rough_crop_pad_ratio']
        bw = dx2 - dx1
        bh = dy2 - dy1
        pad_x = max(int(bw * pad_ratio), 40)
        pad_y = max(int(bh * pad_ratio), 40)

        r1 = max(0, dy1 - pad_y)
        r2 = min(pc_H, dy2 + pad_y)
        c1 = max(0, dx1 - pad_x)
        c2 = min(pc_W, dx2 + pad_x)

        return r1, r2, c1, c2

    # ----------------------------------------------------------------
    #  核心处理管线
    # ----------------------------------------------------------------
    def _process(self, msg: PointCloud2, bbox, det_score=0.0, det_stamp_sec=0.0, det_recv_mono=0.0):
        p = self._p
        frame_id = self._get_frame_id()
        t_total0 = time.monotonic()
        t_stages = {}
        cloud_stamp_sec = stamp_to_sec(msg.header.stamp)
        if det_stamp_sec > 0.0 and cloud_stamp_sec > 0.0:
            t_stages['det_cloud_dt'] = abs(cloud_stamp_sec - det_stamp_sec) * 1000.0
        else:
            t_stages['det_cloud_dt'] = -1.0
        t_stages['det_to_pose'] = (time.monotonic() - det_recv_mono) * 1000.0 if det_recv_mono > 0.0 else -1.0
        self._viz_frame_count += 1
        publish_viz_this_frame = (self._viz_frame_count % self._viz_publish_every_n == 0)

        # ============================================================
        #  1. ★ 两级裁剪: 粗裁剪 (像素) → 精裁剪 (投影)
        # ============================================================
        t1 = time.monotonic()

        if msg.height > 1:
            # ---- 组织化点云: 粗裁剪 (像素) 后再按内参投影精裁剪 ----
            r1, r2, c1, c2 = self._compute_rough_roi(msg, bbox)
            rough_pts = pc2_roi_numpy(msg, r1, r2, c1, c2)
            rough_n = len(rough_pts)
            scene_np = self._crop_by_projection(rough_pts, bbox)
        else:
            # ---- 非组织化: 直接投影裁剪 ----
            all_points = pc2_to_numpy(msg)
            rough_n = len(all_points)
            scene_np = self._crop_by_projection(all_points, bbox)

        t_stages['crop'] = (time.monotonic() - t1) * 1000
        # 首帧诊断
        if not self._first_frame_logged:
            self._first_frame_logged = True
            total_pts = msg.width * msg.height
            self.get_logger().info(
                f"\n{'━'*55}\n"
                f"  ★ 首帧诊断\n"
                f"{'━'*55}\n"
                f"  点云: {msg.width}×{msg.height} = {total_pts} 点\n"
                f"  点云 frame: '{msg.header.frame_id}'\n"
                f"  YOLO 图像: {self._yolo_img_width}×{self._yolo_img_height}\n"
                f"  彩色内参: fx={self._color_fx:.1f} fy={self._color_fy:.1f} "
                f"cx={self._color_cx:.1f} cy={self._color_cy:.1f}\n"
                f"  YOLO bbox: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})\n"
                f"  粗裁剪: {total_pts} → {rough_n} 点\n"
                f"  精裁剪: {rough_n} → {len(scene_np)} 点\n"
                f"{'━'*55}")

        if p['timing_log_enable'] and (not hasattr(self, '_last_roi_diag_mono') or time.monotonic() - self._last_roi_diag_mono > 1.0):
            self._last_roi_diag_mono = time.monotonic()
            self.get_logger().info(f"[ROI DIAG] projected_points={len(scene_np)}")

        if len(scene_np) < p['min_scene_points']:
            self.get_logger().warn(
                f"ROI 点数 {len(scene_np)} 不足, 跳过",
                throttle_duration_sec=2.0)
            return

        # ============================================================
        #  2. ROI 单次体素降采样
        # ============================================================
        t2 = time.monotonic()

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(scene_np.astype(np.float64))

        scene = scene.voxel_down_sample(float(p['scene_voxel_size']))

        n_pts = len(scene.points)
        if p['publish_roi_cloud'] and publish_viz_this_frame:
            self._publish_cloud_np(np.asarray(scene.points), msg.header, self.pub_roi)
        if p['timing_log_enable'] and (not hasattr(self, '_last_preproc_diag_mono') or time.monotonic() - self._last_preproc_diag_mono > 1.0):
            self._last_preproc_diag_mono = time.monotonic()
            self.get_logger().info(
                f"[PREPROC DIAG] scene voxel={p['scene_voxel_size']:.4f}m: "
                f"{len(scene_np)} -> {n_pts} points")
        t_stages['preproc'] = (time.monotonic() - t2) * 1000

        if n_pts < p['min_scene_points']:
            self.get_logger().warn(
                f"预处理后点数 {n_pts} 不足, 跳过",
                throttle_duration_sec=2.0)
            return

        # ============================================================
        #  3. 粗配准 / 精配准 / 追踪
        # ============================================================
        t3 = time.monotonic()
        T, coverage, median = self._register(scene)
        t_stages['register'] = (time.monotonic() - t3) * 1000
        t_stages.update(getattr(self, '_last_reg_timing', {}))

        # 蓝色只表示本次真正送入精配准的 RANSAC 初值。
        if self._last_coarse_transform is not None:
            self._publish_coarse_result(msg.header, self._last_coarse_transform, frame_id)

        if T is None:
            t_stages['total'] = (time.monotonic() - t_total0) * 1000
            self._log_timing(t_stages, det_score, n_pts, coverage, median)
            return

        rot_matrix = np.array(T[:3, :3], dtype=np.float64)
        rot = R.from_matrix(rot_matrix)
        quat = rot.as_quat()    # [x,y,z,w]
        publish_quat = self._quat_with_configured_yaw(quat)
        trans = np.array(T[:3, 3], dtype=np.float64)

        # ============================================================
        #  4. 发布
        # ============================================================
        self._publish_pose(msg.header, trans, publish_quat, frame_id)
        self._publish_tf(msg.header, trans, publish_quat, frame_id)
        if p['publish_model_marker'] and publish_viz_this_frame:
            self._publish_model_marker(msg.header, trans, quat, frame_id)

        t_stages['total'] = (time.monotonic() - t_total0) * 1000
        self._log_timing(t_stages, det_score, n_pts, coverage, median)

    def _log_timing(self, t, det_score, n_pts, coverage, median):
        if not self._p['timing_log_enable']:
            return
        now = time.monotonic()
        throttle = float(self._p['timing_log_throttle_sec'])
        if now - self._last_timing_log_mono < throttle:
            return
        self._last_timing_log_mono = now

        mode = str(t.get('reg_mode', 'TRACK' if self._tracking else 'GLOBAL'))
        dt = t.get('det_cloud_dt', -1.0)
        det_to_pose = t.get('det_to_pose', -1.0)
        dt_txt = f"{dt:.0f}ms" if dt >= 0 else "n/a"
        det_pose_txt = f"{det_to_pose:.0f}ms" if det_to_pose >= 0 else "n/a"

        coarse_txt = ""
        diag = self._last_ransac_diag
        if diag is not None:
            coarse_txt = (
                f" | COARSE corr={diag['correspondences']} "
                f"coverage={diag['coverage']:.3f} median={diag['median']:.4f}m "
                f"feature={diag['feature_ms']:.0f}ms "
                f"ransac={diag['ransac_ms']:.0f}ms")

        icp_txt = ""
        diag = self._last_icp_diag
        if diag is not None:
            icp_txt = (
                f" | ICP {'use' if diag['accepted'] else 'rollback'} "
                f"delta={diag['translation'] * 1000.0:.0f}mm/"
                f"{diag['axis_deg']:.1f}deg "
                f"coverage={diag['initial']['coverage']:.3f}->"
                f"{diag['candidate']['coverage']:.3f} "
                f"median={diag['initial']['median']:.4f}->"
                f"{diag['candidate']['median']:.4f}m")

        median_txt = f"{median:.4f}m" if np.isfinite(median) else "inf"
        self.get_logger().info(
            f"[TIMING {mode}] total={t.get('total', 0):.0f}ms | "
            f"crop={t.get('crop', 0):.0f} "
            f"preproc={t.get('preproc', 0):.0f} "
            f"register={t.get('register', 0):.0f} "
            f"global_coarse={t.get('global_coarse', 0):.0f} "
            f"icp={t.get('icp_fine', 0):.0f}ms"
            f"{coarse_txt}{icp_txt} | "
            f"det_score={det_score:.2f} det-cloud={dt_txt} "
            f"det-to-pose={det_pose_txt} | pts={n_pts} "
            f"scene_coverage={coverage:.3f} scene_median={median_txt}")

    # ----------------------------------------------------------------
    #  Registration
    # ----------------------------------------------------------------
    def _register(self, scene):
        p = self._p
        self._last_reg_timing = {
            'reg_mode': 'TRACK', 'global_coarse': 0.0, 'icp_fine': 0.0}
        self._last_coarse_transform = None
        self._last_ransac_diag = None
        self._last_icp_diag = None

        # Tracking starts from the previous accepted pose and does not compute
        # normals or FPFH.
        if self._tracking and self._last_transform is not None:
            t_icp = time.monotonic()
            transform, metrics = self._run_protected_icp(
                scene, self._last_transform, global_mode=False)
            self._last_reg_timing['icp_fine'] = (
                time.monotonic() - t_icp) * 1000.0
            jump_ok, jump_desc = self._pose_jump_ok(
                self._last_transform, transform)
            metrics_ok, metrics_desc = self._metrics_ok(
                metrics, stage='tracking')
            if jump_ok and metrics_ok:
                self._last_transform = transform
                return transform, metrics['coverage'], metrics['median']

            reason = jump_desc if not jump_ok else metrics_desc
            self.get_logger().warn(
                f"追踪验收失败 ({reason})，重新执行全局配准",
                throttle_duration_sec=1.0)
            self._tracking = False
            self._last_transform = None

        if not bool(p['global_ransac_enable']):
            return None, 0.0, float('inf')
        now = time.monotonic()
        retry = float(p['global_ransac_retry_interval_sec'])
        if (self._last_global_ransac_mono > 0.0 and
                now - self._last_global_ransac_mono < retry):
            return None, 0.0, float('inf')
        self._last_global_ransac_mono = now

        self._last_reg_timing['reg_mode'] = 'GLOBAL'
        t_coarse = time.monotonic()

        # Scene normals exist only for the one FPFH computation in global mode.
        _estimate_normals(
            scene, p['normal_radius'], p['normal_max_nn'],
            camera_location=np.zeros(3))
        t_feature = time.monotonic()
        scene_feature = o3d.pipelines.registration.compute_fpfh_feature(
            scene,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(p['feature_radius']),
                max_nn=int(p['feature_max_nn'])))
        feature_ms = (time.monotonic() - t_feature) * 1000.0

        t_ransac = time.monotonic()
        result = self._try_ransac(
            self.model.cloud, scene, self.model.feature, scene_feature)
        ransac_ms = (time.monotonic() - t_ransac) * 1000.0
        initial = np.array(result.transformation, dtype=np.float64)
        initial_metrics = self._scene_metrics(scene, initial)
        self._last_ransac_diag = {
            'correspondences': len(result.correspondence_set),
            'coverage': initial_metrics['coverage'],
            'median': initial_metrics['median'],
            'feature_ms': feature_ms,
            'ransac_ms': ransac_ms,
        }
        self._last_reg_timing['global_coarse'] = (
            time.monotonic() - t_coarse) * 1000.0

        coarse_ok, coarse_desc = self._metrics_ok(
            initial_metrics, stage='coarse')
        if not coarse_ok:
            self.get_logger().warn(
                f"粗配准未通过场景侧覆盖率验收: {coarse_desc}",
                throttle_duration_sec=1.0)
            return None, initial_metrics['coverage'], initial_metrics['median']

        # Only this accepted RANSAC pose is shown in blue and enters ICP.
        self._last_coarse_transform = initial.copy()
        self.get_logger().info(
            f"粗配准进入 Point-to-Point ICP: "
            f"scene_coverage={initial_metrics['coverage']:.3f} "
            f"scene_median={initial_metrics['median']:.4f}m "
            f"corr={len(result.correspondence_set)}")

        t_icp = time.monotonic()
        transform, metrics = self._run_protected_icp(
            scene, initial, global_mode=True)
        self._last_reg_timing['icp_fine'] = (
            time.monotonic() - t_icp) * 1000.0

        final_ok, final_desc = self._metrics_ok(metrics, stage='icp')
        if not final_ok:
            self.get_logger().warn(
                f"精配准未通过场景侧覆盖率验收: {final_desc}",
                throttle_duration_sec=1.0)
            return None, metrics['coverage'], metrics['median']

        self._last_transform = transform
        self._tracking = True
        return transform, metrics['coverage'], metrics['median']

    def _scene_metrics(self, scene, transform):
        """Evaluate scene-to-model coverage; invisible model faces are irrelevant."""
        scene_points = np.asarray(scene.points)
        if len(scene_points) == 0:
            return {'coverage': 0.0, 'median': float('inf')}

        # Rigid distances are invariant: inverse-transforming the scene lets
        # every evaluation reuse the model KD-tree built during initialization.
        rotation = np.asarray(transform[:3, :3])
        translation = np.asarray(transform[:3, 3])
        scene_in_model = (scene_points - translation) @ rotation
        distances, _ = self.model.kdtree.query(scene_in_model, k=1)
        return {
            'coverage': float(np.mean(
                distances < float(self._p['scene_coverage_dist_thresh']))),
            'median': float(np.median(distances)),
        }

    def _metrics_ok(self, metrics, stage):
        if stage == 'coarse':
            min_coverage = float(self._p['coarse_min_scene_coverage'])
            max_median = float(self._p['coarse_max_scene_median'])
        elif stage == 'tracking':
            min_coverage = float(self._p['tracking_min_scene_coverage'])
            max_median = float(self._p['tracking_max_scene_median'])
        else:
            min_coverage = float(self._p['icp_min_scene_coverage'])
            max_median = float(self._p['icp_max_scene_median'])

        ok = (
            metrics['coverage'] >= min_coverage and
            metrics['median'] <= max_median)
        desc = (
            f"coverage={metrics['coverage']:.3f}/{min_coverage:.3f} "
            f"median={metrics['median']:.4f}/{max_median:.4f}m")
        return ok, desc

    @staticmethod
    def _pose_delta(reference, candidate):
        translation = float(np.linalg.norm(
            np.asarray(candidate[:3, 3]) - np.asarray(reference[:3, 3])))
        reference_axis = np.asarray(reference[:3, 2], dtype=np.float64)
        candidate_axis = np.asarray(candidate[:3, 2], dtype=np.float64)
        cosine = np.dot(reference_axis, candidate_axis) / max(
            1e-12,
            np.linalg.norm(reference_axis) * np.linalg.norm(candidate_axis))
        axis_deg = float(np.rad2deg(
            np.arccos(np.clip(cosine, -1.0, 1.0))))
        return translation, axis_deg

    def _pose_jump_ok(self, old, new):
        max_translation = float(self._p['max_tracking_translation_jump'])
        max_rotation = np.deg2rad(
            float(self._p['max_tracking_rotation_jump_deg']))
        translation = float(np.linalg.norm(new[:3, 3] - old[:3, 3]))
        if bool(self._p['axisymmetric_ignore_yaw_jump']):
            old_axis = np.asarray(old[:3, 2], dtype=np.float64)
            new_axis = np.asarray(new[:3, 2], dtype=np.float64)
            cosine = np.dot(old_axis, new_axis) / max(
                1e-12, np.linalg.norm(old_axis) * np.linalg.norm(new_axis))
            rotation = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            label = 'axis'
        else:
            rotation = float(R.from_matrix(
                np.asarray(old[:3, :3]).T @
                np.asarray(new[:3, :3])).magnitude())
            label = 'rotation'
        ok = translation <= max_translation and rotation <= max_rotation
        return (
            ok,
            f"pose_jump translation={translation:.3f}/"
            f"{max_translation:.3f}m {label}="
            f"{np.rad2deg(rotation):.1f}/"
            f"{np.rad2deg(max_rotation):.1f}deg")

    def _try_ransac(self, source, target, source_feature, target_feature):
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                float(self._p['ransac_edge_length_ratio'])),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                float(self._p['ransac_dist_thresh'])),
        ]
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=source,
            target=target,
            source_feature=source_feature,
            target_feature=target_feature,
            mutual_filter=bool(self._p['ransac_mutual_filter']),
            max_correspondence_distance=float(
                self._p['ransac_dist_thresh']),
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False)),
            ransac_n=int(self._p['ransac_n']),
            checkers=checkers,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                int(self._p['ransac_max_iter']),
                float(self._p['ransac_confidence'])))

    def _run_protected_icp(self, scene, initial, global_mode):
        """Run one Point-to-Point ICP and keep the input when it leaves trust."""
        p = self._p
        initial = np.array(initial, dtype=np.float64)
        initial_metrics = self._scene_metrics(scene, initial)

        if global_mode:
            distance = float(p['icp_dist_thresh'])
            iterations = int(p['icp_max_iter'])
            max_translation = float(p['global_icp_max_translation'])
            max_axis_deg = float(p['global_icp_max_axis_deg'])
        else:
            distance = float(p['tracking_icp_dist_thresh'])
            iterations = int(p['tracking_icp_max_iter'])
            max_translation = float(p['max_tracking_translation_jump'])
            max_axis_deg = float(p['max_tracking_rotation_jump_deg'])

        result = o3d.pipelines.registration.registration_icp(
            source=self.model.cloud,
            target=scene,
            max_correspondence_distance=distance,
            init=initial,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False)),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations,
                relative_fitness=float(p['icp_relative_fitness']),
                relative_rmse=float(p['icp_relative_rmse'])))
        candidate = np.array(result.transformation, dtype=np.float64)
        candidate_metrics = self._scene_metrics(scene, candidate)
        translation, axis_deg = self._pose_delta(initial, candidate)

        within_trust = (
            translation <= max_translation and axis_deg <= max_axis_deg)
        quality_not_worse = (
            candidate_metrics['coverage'] >= (
                initial_metrics['coverage'] -
                float(p['icp_max_scene_coverage_drop'])) and
            candidate_metrics['median'] <= (
                initial_metrics['median'] +
                float(p['icp_max_scene_median_increase'])))
        accepted = bool(within_trust and quality_not_worse)
        self._last_icp_diag = {
            'accepted': accepted,
            'translation': translation,
            'axis_deg': axis_deg,
            'initial': initial_metrics,
            'candidate': candidate_metrics,
        }
        if not accepted:
            reason = (
                'trust region' if not within_trust
                else 'scene coverage/median degraded beyond tolerance')
            self.get_logger().warn(
                f"ICP 回退粗姿态/上一帧姿态: {reason}; "
                f"delta={translation * 1000.0:.0f}mm/{axis_deg:.1f}deg, "
                f"coverage={initial_metrics['coverage']:.3f}->"
                f"{candidate_metrics['coverage']:.3f}, "
                f"median={initial_metrics['median']:.4f}->"
                f"{candidate_metrics['median']:.4f}m",
                throttle_duration_sec=0.5)
            return initial, initial_metrics
        return candidate, candidate_metrics

    # ----------------------------------------------------------------
    #  Publishing
    # ----------------------------------------------------------------
    def _quat_with_configured_yaw(self, quat):
        rpy = R.from_quat(quat).as_euler('xyz', degrees=False)
        rpy[2] = float(self._p.get('pose_yaw', 0.0))
        return R.from_euler('xyz', rpy, degrees=False).as_quat()

    def _publish_pose_msg(self, header, trans, quat, frame_id, publisher):
        msg = PoseStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        publisher.publish(msg)

    def _publish_pose(self, header, trans, quat, frame_id):
        self._publish_pose_msg(header, trans, quat, frame_id, self.pub_pose)

    def _publish_tf(self, header, trans, quat, frame_id):
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = frame_id
        t.child_frame_id = 'object_pose'
        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)

    def _publish_model_marker_core(
            self, header, trans, quat, frame_id, publisher, color, pts,
            marker_ns='model', marker_id=0, marker_scale=0.002):
        m = Marker()
        m.header.stamp = header.stamp
        m.header.frame_id = frame_id
        m.ns = marker_ns
        m.id = int(marker_id)
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = float(marker_scale)
        m.scale.y = float(marker_scale)
        m.color = color

        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = trans
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_t = (T @ pts_h.T).T[:, :3]

        # 每隔 3 个点取一个, 减少 Marker 负载
        m.points = [Point(x=float(r[0]), y=float(r[1]), z=float(r[2]))
                     for r in pts_t[::3]]
        publisher.publish(m)

    def _publish_model_marker(self, header, trans, quat, frame_id):
        """Publish the final ICP or tracking result in green."""
        self._publish_model_marker_core(
            header, trans, quat, frame_id, self.pub_marker,
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
            np.asarray(self.model.cloud.points),
            marker_ns='final', marker_id=0)

    def _publish_coarse_marker(self, header, trans, quat, frame_id):
        """Publish the accepted RANSAC pose that actually enters ICP in blue."""
        self._publish_model_marker_core(
            header, trans, quat, frame_id, self.pub_coarse_marker,
            ColorRGBA(r=0.0, g=0.2, b=1.0, a=0.8),
            np.asarray(self.model.cloud.points),
            marker_ns='coarse', marker_id=0)

    def _publish_coarse_result(self, header, transform, frame_id):
        rotation = R.from_matrix(
            np.asarray(transform[:3, :3], dtype=np.float64))
        quat = rotation.as_quat()
        translation = np.asarray(transform[:3, 3], dtype=np.float64)
        if self._p['publish_coarse_marker']:
            self._publish_coarse_marker(
                header, translation, quat, frame_id)
        if self._p['publish_coarse_pose']:
            self._publish_pose_msg(
                header, translation, quat, frame_id, self.pub_coarse_pose)

    def _publish_cloud_np(self, pts, header, publisher):
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        n = len(pts)
        msg = PointCloud2()
        msg.header = copy.deepcopy(header)
        msg.height = 1
        msg.width = n
        msg.fields = [
            _make_field('x', 0), _make_field('y', 4), _make_field('z', 8)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n
        msg.is_dense = True
        msg.data = pts.tobytes()
        publisher.publish(msg)

    # ----------------------------------------------------------------
    #  服务
    # ----------------------------------------------------------------
    def _srv_reset(self, req, resp):
        with self._lock:
            self._tracking = False
            self._last_transform = None
        resp.success = True
        resp.message = "追踪已重置"
        self.get_logger().info(resp.message)
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
