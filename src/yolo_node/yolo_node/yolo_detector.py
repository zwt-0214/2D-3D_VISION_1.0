#!/usr/bin/env python3
"""
ROS2 YOLO Detection Node using OpenVINO
订阅 D435i 相机 RGB 图像，使用 YOLO11n + OpenVINO 进行目标检测
"""

import os
import sys
from pathlib import Path


def _maybe_reexec_with_target_python() -> None:
    """Re-exec with conda Python when launched by system Python."""
    target_python = os.environ.get(
        'YOLO_NODE_PYTHON',
        '/home/zwt/miniconda3/envs/foundationpose/bin/python',
    )
    try:
        target = Path(target_python)
        if not target.exists():
            return
        if os.path.realpath(sys.executable) == os.path.realpath(str(target)):
            return
        os.execv(
            str(target),
            [
                str(target),
                '-m',
                'yolo_node.yolo_detector',
                *sys.argv[1:],
            ],
        )
    except Exception:
        return


_maybe_reexec_with_target_python()


def _ensure_ultralytics_path() -> None:
    candidates = []
    env_root = os.environ.get('ULTRALYTICS_ROOT')
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        Path('/home/zwt/vision/src/ultralytics'),
        Path('/home/zwt/vision/src'),
    ])

    for p in candidates:
        try:
            if not p.exists():
                continue
            # repo root like /home/zwt/vision/src/ultralytics
            if (p / 'ultralytics').exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                return
            # direct package dir fallback
            if p.name == 'ultralytics' and (p / '__init__.py').exists():
                parent = p.parent
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                return
        except Exception:
            continue


_ensure_ultralytics_path()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import tempfile
import shutil
import traceback
import threading
import time
from collections import defaultdict, deque


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# Ultralytics for YOLO model
from ultralytics import YOLO


def _reshape_image(data: np.ndarray, msg: Image, channels: int) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image shape")

    elem = data.dtype.itemsize
    step = int(msg.step) if msg.step else w * channels * elem
    row_elems = step // elem

    if channels == 1:
        if row_elems == w:
            return data.reshape(h, w)
        return data.reshape(h, row_elems)[:, :w]

    if row_elems == w * channels:
        return data.reshape(h, w, channels)
    return data.reshape(h, row_elems)[:, : w * channels].reshape(h, w, channels)


def _image_msg_to_bgr(msg: Image) -> np.ndarray:
    enc = (msg.encoding or "").lower()
    if enc not in ("bgr8", "rgb8", "mono8"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    dtype = np.dtype(np.uint8)
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")

    data = np.frombuffer(msg.data, dtype=dtype)
    if enc in ("bgr8", "rgb8"):
        img = _reshape_image(data, msg, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    gray = _reshape_image(data, msg, 1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _bgr_to_image_msg(img: np.ndarray, header) -> Image:
    if img.dtype != np.uint8:
        raise ValueError("Expected uint8 image for bgr8 encoding")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 image for bgr8 encoding")

    out = Image()
    out.header = header
    out.height = int(img.shape[0])
    out.width = int(img.shape[1])
    out.encoding = "bgr8"
    out.is_bigendian = False
    out.step = int(img.shape[1] * 3)
    out.data = np.ascontiguousarray(img).tobytes()
    return out


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        # 声明参数
        self.declare_parameter('model_path', '/home/zwt/vision/ultralytics/results/yolo11n2/weights/openvino')
        self.declare_parameter('conf_threshold', 0.78)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('publish_conf_threshold', 0.78)
        self.declare_parameter('publish_hold_seconds', 0.0)
        self.declare_parameter('device', 'cpu')  # 'cpu' or 'gpu' for OpenVINO
        # self.declare_parameter('image_topic', 'camera/camera/color/image_raw')
        self.declare_parameter('image_topic', '/camera/color/image_raw')

        self.declare_parameter('detection_topic', 'yolo/detections')
        self.declare_parameter('annotated_image_topic', 'yolo/annotated_image')
        self.declare_parameter('left_or_right', 0)
        self.declare_parameter('timing_log_enable', True)
        self.declare_parameter('timing_log_throttle_sec', 0.5)
        self.declare_parameter('image_queue_depth', 2)
        self.declare_parameter('NUC', False)
        self.declare_parameter('yolo_threads_default', 4)
        self.declare_parameter('yolo_threads_nuc', 6)
        self.declare_parameter('bbox_area_smoothing_enable', True)
        self.declare_parameter('bbox_area_history_len', 10)
        self.declare_parameter('bbox_area_min_history', 3)
        self.declare_parameter('bbox_area_jump_ratio', 0.55)
        
        # 获取参数
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.publish_conf_threshold = self.get_parameter('publish_conf_threshold').value
        self.publish_hold_seconds = self.get_parameter('publish_hold_seconds').value
        device = self.get_parameter('device').value
        self.device = device
        image_topic = self.get_parameter('image_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        annotated_image_topic = self.get_parameter('annotated_image_topic').value
        self.left_or_right = int(self.get_parameter('left_or_right').value)
        self.timing_log_enable = bool(self.get_parameter('timing_log_enable').value)
        self.timing_log_throttle_sec = float(self.get_parameter('timing_log_throttle_sec').value)
        self.image_queue_depth = max(1, int(self.get_parameter('image_queue_depth').value))
        self.NUC = _as_bool(self.get_parameter('NUC').value)
        self.yolo_threads_default = max(1, int(self.get_parameter('yolo_threads_default').value))
        self.yolo_threads_nuc = max(1, int(self.get_parameter('yolo_threads_nuc').value))
        self.bbox_area_smoothing_enable = _as_bool(self.get_parameter('bbox_area_smoothing_enable').value)
        self.bbox_area_history_len = max(1, int(self.get_parameter('bbox_area_history_len').value))
        self.bbox_area_min_history = max(1, int(self.get_parameter('bbox_area_min_history').value))
        self.bbox_area_jump_ratio = max(0.05, float(self.get_parameter('bbox_area_jump_ratio').value))
        self._last_timing_log_mono = 0.0
        self._haomu_area_histories = defaultdict(lambda: deque(maxlen=self.bbox_area_history_len))
        self._last_area_jump_log_mono = 0.0
        self._apply_compute_profile()
        
        # CV Bridge 用于图像转换
        self.bridge = CvBridge()
        self._cv_bridge_ok = True

        # 发布门控状态：仅当高置信度持续满足一段时间后才发布
        self.high_conf_start_time_sec: float | None = None
        self.publish_gate_open = False
        
        # 仅加载 OpenVINO 模型
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        try:
            import importlib
            ultralytics = importlib.import_module('ultralytics')
            self.get_logger().info(f"Ultralytics version: {ultralytics.__version__}")
        except Exception:
            pass
        self.model = self._load_model(model_path, device)
        self.get_logger().info('YOLO model loaded successfully!')
        
        # 延迟导入自定义消息 (编译后才能导入)
        from yolo_node.msg import Detection, DetectionArray
        self.Detection = Detection
        self.DetectionArray = DetectionArray
        
        # 创建订阅者 - 订阅 D435i 相机 RGB 话题
        # KEEP_LAST=2 + 后台线程覆盖 latest，可避免推理慢于相机帧率时旧图像堆积，同时保留极小缓冲。
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.image_queue_depth,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            image_qos
        )
        self.get_logger().info(f'Subscribed to: {image_topic} (KEEP_LAST={self.image_queue_depth}, latest-frame only)')
        
        # 创建发布者 - 发布检测结果
        self.detection_publisher = self.create_publisher(
            DetectionArray,
            detection_topic,
            3
        )
        self.get_logger().info(f'Publishing detections to: {detection_topic}')
        
        # 创建发布者 - 发布带标注的图像 (可选)
        self.annotated_image_publisher = self.create_publisher(
            Image,
            annotated_image_topic,
            3
        )
        self.get_logger().info(f'Publishing annotated images to: {annotated_image_topic}')

        self.get_logger().info(
            f'Publish gating enabled: conf>{self.publish_conf_threshold} '
            f'for {self.publish_hold_seconds:.2f}s before publishing'
        )
        self.get_logger().info(
            f'haomu selection mode enabled: left_or_right={self.left_or_right} '
            f'(0=left, 1=right)'
        )
        
        self.get_logger().info('YOLO Detector Node initialized successfully!')

        # 异步推理：仅保留最新帧，避免旧帧排队导致输出滞后
        self._image_lock = threading.Lock()
        self._image_cv = threading.Condition(self._image_lock)
        self._latest_image_item: tuple[Image, float] | None = None
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._inference_worker,
            name='yolo_inference_worker',
            daemon=True,
        )
        self._worker_thread.start()
    

    def _apply_compute_profile(self):
        """Limit YOLO/OpenVINO CPU usage according to platform profile."""
        threads = self.yolo_threads_nuc if self.NUC else self.yolo_threads_default
        threads = max(1, int(threads))
        os.environ['NUC'] = 'true' if self.NUC else 'false'
        os.environ['YOLO_COMPUTE_PROFILE'] = 'nuc' if self.NUC else 'jiaolong16q'
        os.environ['OV_CPU_THREADS_NUM'] = str(threads)
        os.environ['OPENVINO_CPU_THREADS_NUM'] = str(threads)
        os.environ.setdefault('OMP_NUM_THREADS', str(threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(threads))
        profile = 'NUC12 i7 12C/20T' if self.NUC else '机械革命蛟龙16Q 8C/16T'
        self.get_logger().info(f'YOLO compute profile: {profile}, OpenVINO/OMP threads≈{threads}')

    def _stable_haomu_area(self, det, slot: int) -> float:
        """Return history-averaged area for ranking to suppress one-frame box size spikes."""
        area = max(1.0, float(det.width) * float(det.height))
        hist = self._haomu_area_histories[int(slot)]
        if not self.bbox_area_smoothing_enable:
            hist.append(area)
            return area

        if len(hist) >= self.bbox_area_min_history:
            prev_mean = float(np.mean(hist))
            jump_ratio = abs(area - prev_mean) / max(prev_mean, 1.0)
            if jump_ratio > self.bbox_area_jump_ratio:
                now = time.monotonic()
                if now - self._last_area_jump_log_mono > 0.8:
                    self._last_area_jump_log_mono = now
                    self.get_logger().warn(
                        f'haomu bbox area spike suppressed: slot={slot} area={area:.0f} '
                        f'mean={prev_mean:.0f} jump={jump_ratio:.2f}'
                    )
                return prev_mean

        hist.append(area)
        return float(np.mean(hist))


    def _select_haomu_detection(self, detections: list):
        """
        当检测到多个 haomu 时：
        1. 按 x 坐标为 haomu 检测分配稳定槽位；
        2. 用最近 bbox_area_history_len 帧面积均值参与“面积前二”筛选，抑制单帧框大小突变；
        3. 再根据 left_or_right 选择偏左(0)或偏右(1)的那个。

        其余类别保持不变；若 haomu 数量不足 2，则保持原有结果。
        """
        haomu_detections = [det for det in detections if det.class_name == 'haomu']
        if len(haomu_detections) < 2:
            if len(haomu_detections) == 1:
                self._stable_haomu_area(haomu_detections[0], 0)
            return detections

        by_x = sorted(haomu_detections, key=lambda det: float(det.x) + 0.5 * float(det.width))
        stable_area = {}
        for slot, det in enumerate(by_x):
            stable_area[id(det)] = self._stable_haomu_area(det, slot)

        top2 = sorted(
            haomu_detections,
            key=lambda det: stable_area.get(id(det), float(det.width) * float(det.height)),
            reverse=True,
        )[:2]

        choose_right = 1 if int(self.left_or_right) == 1 else 0
        selected_haomu = sorted(
            top2,
            key=lambda det: float(det.x) + 0.5 * float(det.width),
        )[-1 if choose_right else 0]

        filtered = [det for det in detections if det.class_name != 'haomu']
        filtered.append(selected_haomu)
        return filtered

    def _load_model(self, model_path: str, device: str) -> YOLO:
        """
        仅加载 OpenVINO 格式模型目录（不读取 .pt，不做导出）
        """
        model_path = Path(model_path)

        if not model_path.exists():
            self.get_logger().error(f'OpenVINO model path not found: {model_path}')
            raise FileNotFoundError(f'OpenVINO model path not found: {model_path}')

        def _prepare_openvino_dir_alias(src_dir: Path) -> Path | None:
            """为不符合 *_openvino_model 命名的目录创建兼容别名目录。"""
            if not src_dir.is_dir():
                return None

            xml_files_local = sorted(src_dir.glob('*.xml'))
            bin_files_local = sorted(src_dir.glob('*.bin'))
            if not xml_files_local or not bin_files_local:
                return None

            if src_dir.name.endswith('_openvino_model'):
                return src_dir

            xml_stem = xml_files_local[0].stem
            alias_root = Path(tempfile.gettempdir()) / 'yolo_node_openvino_alias'
            alias_root.mkdir(parents=True, exist_ok=True)
            alias_dir = alias_root / f'{xml_stem}_openvino_model'

            if alias_dir.exists():
                try:
                    if alias_dir.is_symlink() or alias_dir.is_file():
                        alias_dir.unlink()
                    elif alias_dir.is_dir():
                        shutil.rmtree(alias_dir)
                except Exception:
                    pass

            # 优先软链接，失败则复制
            try:
                alias_dir.symlink_to(src_dir, target_is_directory=True)
                self.get_logger().info(f'Created OpenVINO alias symlink: {alias_dir} -> {src_dir}')
            except Exception:
                shutil.copytree(src_dir, alias_dir)
                self.get_logger().info(f'Created OpenVINO alias copy: {alias_dir} (from {src_dir})')

            return alias_dir

        # 兼容不同 Ultralytics 版本：有的支持目录，有的更偏向 xml
        candidates: list[Path] = [model_path]
        alias_dir = _prepare_openvino_dir_alias(model_path)
        if alias_dir is not None and alias_dir != model_path:
            candidates.insert(0, alias_dir)
        xml_files = sorted(model_path.glob('*.xml')) if model_path.is_dir() else []
        candidates.extend(xml_files)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                self.get_logger().info(f'Trying OpenVINO model: {candidate}')
                loaded = YOLO(str(candidate), task='detect')
                # 提前触发后端初始化，尽早暴露加载问题
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = loaded.predict(source=dummy, device=device, verbose=False)
                return loaded
            except Exception as e:
                last_error = e
                self.get_logger().warning(f'Failed to load OpenVINO model from {candidate}: {e}')

        raise RuntimeError(
            f'Unable to load OpenVINO model from {model_path}. Last error: {last_error}'
        )
    
    def image_callback(self, msg: Image):
        """
        图像回调函数 - 仅缓存最新图像，实际推理在后台线程执行
        """
        with self._image_cv:
            self._latest_image_item = (msg, time.monotonic())
            self._image_cv.notify()

    def _inference_worker(self):
        while self._worker_running:
            with self._image_cv:
                while self._worker_running and self._latest_image_item is None:
                    self._image_cv.wait(timeout=0.1)
                if not self._worker_running:
                    return
                item = self._latest_image_item
                self._latest_image_item = None

            if item is None:
                continue
            msg, recv_mono = item
            self._process_image(msg, recv_mono)

    def _process_image(self, msg: Image, recv_mono: float = 0.0):
        """
        图像回调函数 - 处理接收到的 RGB 图像
        """
        try:
            t_total0 = time.monotonic()
            t_convert0 = time.monotonic()
            queue_delay_ms = (t_total0 - recv_mono) * 1000.0 if recv_mono > 0.0 else 0.0
            # 将 ROS Image 消息转换为 OpenCV 格式
            if self._cv_bridge_ok:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except Exception as e:
                    self._cv_bridge_ok = False
                    self.get_logger().error(
                        f'cv_bridge failed ({e}); falling back to raw conversion'
                    )
                    cv_image = _image_msg_to_bgr(msg)
            else:
                cv_image = _image_msg_to_bgr(msg)
            
            convert_ms = (time.monotonic() - t_convert0) * 1000.0

            # 运行 YOLO 推理
            t_infer0 = time.monotonic()
            results = self.model.predict(
                source=cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            infer_ms = (time.monotonic() - t_infer0) * 1000.0
            
            # 创建检测结果消息
            detection_array_msg = self.DetectionArray()
            detection_array_msg.header = msg.header
            max_confidence = 0.0
            
            t_post0 = time.monotonic()

            # 解析检测结果
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        detection_msg = self.Detection()
                        
                        # 边界框坐标 (xyxy 格式转为 xywh)
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        detection_msg.x = float(xyxy[0])
                        detection_msg.y = float(xyxy[1])
                        detection_msg.width = float(xyxy[2] - xyxy[0])
                        detection_msg.height = float(xyxy[3] - xyxy[1])
                        
                        # 置信度
                        confidence = float(boxes.conf[i].cpu().numpy())
                        detection_msg.confidence = confidence
                        if confidence > max_confidence:
                            max_confidence = confidence
                        
                        # 类别 ID 和名称
                        class_id = int(boxes.cls[i].cpu().numpy())
                        detection_msg.class_id = class_id
                        detection_msg.class_name = self.model.names.get(class_id, f'class_{class_id}')
                        
                        detection_array_msg.detections.append(detection_msg)
            
            # 对 haomu 做后处理：多个同类目标时，仅保留按规则选中的一个
            selected_detections = self._select_haomu_detection(detection_array_msg.detections)
            if len(selected_detections) != len(detection_array_msg.detections):
                detection_array_msg.detections = selected_detections
                max_confidence = max((det.confidence for det in selected_detections), default=0.0)

            # 发布门控：仅当 max_confidence 持续超过阈值指定时长后才允许发布
            now_sec = self.get_clock().now().nanoseconds / 1e9
            should_publish = False

            if max_confidence > self.publish_conf_threshold:
                if self.high_conf_start_time_sec is None:
                    self.high_conf_start_time_sec = now_sec
                    self.get_logger().info(
                        f'High confidence detected (max={max_confidence:.3f}), start hold timer...'
                    )

                hold_elapsed = now_sec - self.high_conf_start_time_sec
                if hold_elapsed >= self.publish_hold_seconds:
                    should_publish = True
                    if not self.publish_gate_open:
                        self.publish_gate_open = True
                        self.get_logger().info(
                            f'Publish gate opened: confidence>{self.publish_conf_threshold} '
                            f'held for {hold_elapsed:.3f}s. Publishing enabled.'
                        )
            else:
                if self.high_conf_start_time_sec is not None or self.publish_gate_open:
                    self.get_logger().info(
                        f'Publish gate reset (max_conf={max_confidence:.3f} <= '
                        f'{self.publish_conf_threshold}). Publishing paused.'
                    )
                self.high_conf_start_time_sec = None
                self.publish_gate_open = False

            post_ms = (time.monotonic() - t_post0) * 1000.0

            if should_publish:
                # 发布检测结果
                self.detection_publisher.publish(detection_array_msg)

                # 发布带标注的图像
                if self.annotated_image_publisher.get_subscription_count() > 0:
                    annotated_frame = cv_image.copy()
                    for det in detection_array_msg.detections:
                        x1 = int(round(det.x))
                        y1 = int(round(det.y))
                        x2 = int(round(det.x + det.width))
                        y2 = int(round(det.y + det.height))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det.class_name} {det.confidence:.2f}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    if self._cv_bridge_ok:
                        annotated_msg = self.bridge.cv2_to_imgmsg(
                            annotated_frame, encoding='bgr8'
                        )
                        annotated_msg.header = msg.header
                    else:
                        annotated_msg = _bgr_to_image_msg(annotated_frame, msg.header)
                    self.annotated_image_publisher.publish(annotated_msg)
            
            # 终端计时日志
            total_ms = (time.monotonic() - t_total0) * 1000.0
            self._log_timing(
                queue_delay_ms=queue_delay_ms,
                convert_ms=convert_ms,
                infer_ms=infer_ms,
                post_ms=post_ms,
                total_ms=total_ms,
                num_detections=len(detection_array_msg.detections),
                max_confidence=max_confidence,
                should_publish=should_publish)

            # 日志输出检测数量
            num_detections = len(detection_array_msg.detections)
            if num_detections > 0:
                self.get_logger().debug(f'Detected {num_detections} objects')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def _log_timing(self, queue_delay_ms, convert_ms, infer_ms, post_ms, total_ms,
                    num_detections, max_confidence, should_publish):
        if not self.timing_log_enable:
            return
        now = time.monotonic()
        if now - self._last_timing_log_mono < self.timing_log_throttle_sec:
            return
        self._last_timing_log_mono = now
        self.get_logger().info(
            f"[YOLO TIMING] total={total_ms:.0f}ms | queue={queue_delay_ms:.0f} "
            f"convert={convert_ms:.0f} infer={infer_ms:.0f} post={post_ms:.0f}ms | "
            f"det={num_detections} max_conf={max_confidence:.2f} publish={should_publish}")

    def destroy_node(self):
        self._worker_running = False
        with self._image_cv:
            self._image_cv.notify_all()
        if hasattr(self, '_worker_thread'):
            self._worker_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = YoloDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
