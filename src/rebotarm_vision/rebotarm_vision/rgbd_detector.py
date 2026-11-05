"""MuJoCo 与眼在手上的 RealSense D435 共用的 RGB-D 检测节点。"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .rgbd_detection import detect_blocks, detect_pingpong_ball
from .yolo_detection import detect_yolo_blocks, parse_class_map


class RgbdDetector(Node):
    def __init__(self):
        super().__init__("rebotarm_rgbd_detector")
        defaults = {
            "arm_namespace": "rebotarm",
            "base_frame": "base_link",
            "use_pose_topic": False,
            "image_topic": "/rebotarm/mujoco/d435/image_raw",
            "depth_topic": "/rebotarm/mujoco/d435/depth",
            "camera_info_topic": "/rebotarm/mujoco/d435/camera_info",
            "pose_topic": "/rebotarm/mujoco/d435/pose",
            "depth_scale_16uc1": 0.001,
            "transform_timeout_sec": 0.15,
            "detection_source": "camera_rgbd",
            # 抓取和直接运行节点都默认使用 YOLO；颜色分割只能显式备用。
            "detector_backend": "yolo",
            # 相对路径按启动目录解析；训练完成后将权重放到工作空间 models/best.pt。
            "yolo_model": "models/best.pt",
            "yolo_confidence": 0.45,
            "yolo_iou": 0.50,
            # 推理缩放尺寸；小目标可适当调大，显存占用和推理耗时也会增加。
            "yolo_imgsz": 640,
            # 0 为第一块 NVIDIA GPU；可显式传 cpu 进行无显卡调试。
            "yolo_device": "0",
            # 把任意模型类别映射到当前抓取配置，例如：
            # '{"red_cube":"red","blue_cup":"blue"}'。
            "yolo_class_map": "{}",
        }
        for name, value in defaults.items():
            # ROS 命令行会把 yolo_device:=0 推断为整数；允许它同时接收
            # 整数 GPU 编号、"cuda:0" 和 "cpu" 字符串。
            if name == "yolo_device":
                self.declare_parameter(
                    name, value, ParameterDescriptor(dynamic_typing=True)
                )
            else:
                self.declare_parameter(name, value)
        get = lambda name: self.get_parameter(name).value
        self.namespace = str(get("arm_namespace")).strip("/") or "rebotarm"
        self.base_frame = str(get("base_frame"))
        self.use_pose_topic = bool(get("use_pose_topic"))
        self.depth_scale = float(get("depth_scale_16uc1"))
        self.tf_timeout = float(get("transform_timeout_sec"))
        self.detector_backend = str(get("detector_backend")).strip().lower() or "yolo"
        self.yolo = None
        if self.detector_backend not in ("color", "yolo"):
            raise ValueError("detector_backend 必须是 color 或 yolo")
        self.detection_source = str(get("detection_source")).strip() or "camera_rgbd"
        if self.detector_backend == "yolo" and self.detection_source == "camera_rgbd":
            self.detection_source = "yolo_rgbd"
        self.yolo_confidence = float(get("yolo_confidence"))
        self.yolo_iou = float(get("yolo_iou"))
        self.yolo_imgsz = int(get("yolo_imgsz"))
        self.yolo_device = str(get("yolo_device")).strip() or "0"
        # 启动时就校验类别映射，避免相机开始发图后才发现 JSON 写错。
        try:
            self.yolo_class_map = parse_class_map(get("yolo_class_map"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "yolo_class_map 必须是 JSON 对象，例如 "
                "'{\"red_cube\":\"red\"}'"
            ) from exc
        if self.detector_backend == "yolo":
            if self.yolo_imgsz <= 0:
                raise ValueError("yolo_imgsz 必须大于 0")
            model_value = str(get("yolo_model")).strip()
            if not model_value:
                raise ValueError("使用 YOLO 时必须提供 yolo_model:=模型权重路径")
            model_path = Path(model_value).expanduser().resolve()
            # 只读取用户指定的本地权重，避免缺少 best.pt 时加载其他模型。
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"YOLO 权重不存在：{model_path}。训练后放入 models/best.pt，"
                    "或通过 yolo_model:=/绝对路径/best.pt 指定。"
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "使用 YOLO 需要先安装可选依赖 ultralytics"
                ) from exc
            self.yolo = YOLO(str(model_path))
            # 启动时预热并检查实际 CUDA 推理，避免第一张相机图像才暴露环境问题。
            # 不自动改用 CPU；显式指定 cpu 时才使用 CPU。
            try:
                self.yolo.predict(
                    np.zeros((64, 64, 3), dtype=np.uint8), imgsz=self.yolo_imgsz,
                    device=self.yolo_device, verbose=False,
                )
            except (RuntimeError, ValueError, AssertionError) as exc:
                raise RuntimeError(
                    f"YOLO 推理初始化失败（device={self.yolo_device}）：{exc}。"
                    "请检查运行 ROS 的 Python 环境中的 torch/torchvision 与 CUDA；"
                    "临时使用 CPU 可设置 yolo_device:=cpu。"
                ) from exc
            predictor_device = getattr(getattr(self.yolo, "predictor", None), "device", self.yolo_device)
            self.get_logger().info(
                f"YOLO 权重：{model_path}，推理设备：{predictor_device}，"
                f"推理尺寸：{self.yolo_imgsz}"
            )
        self.required = {"rgb", "depth", "info"}
        if self.use_pose_topic:
            self.required.add("pose")
        self.frames: OrderedDict[int, dict] = OrderedDict()
        self.last_stamp = 0
        self.last_frame = time.monotonic()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.block_pub = self.create_publisher(
            String, f"/{self.namespace}/vision/color_blocks/detections", 10
        )
        self.ball_pub = self.create_publisher(
            String, f"/{self.namespace}/vision/pingpong_ball", 10
        )
        subscriptions = (
            ("rgb", Image, str(get("image_topic"))),
            ("depth", Image, str(get("depth_topic"))),
            ("info", CameraInfo, str(get("camera_info_topic"))),
        )
        for kind, cls, topic in subscriptions:
            self.create_subscription(
                cls, topic, lambda msg, key=kind: self._receive(key, msg),
                qos_profile_sensor_data
            )
        if self.use_pose_topic:
            self.create_subscription(
                PoseStamped, str(get("pose_topic")),
                lambda msg: self._receive("pose", msg), qos_profile_sensor_data
            )
        self.create_timer(0.25, self._expire)
        self.get_logger().info(
            f"RGB-D 检测器已启动：namespace=/{self.namespace}, pose_source="
            f"{'topic' if self.use_pose_topic else 'tf2'}, backend={self.detector_backend}"
        )

    @staticmethod
    def _stamp_ns(msg) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _receive(self, kind, msg):
        stamp = self._stamp_ns(msg)
        if stamp <= self.last_stamp:
            return
        frame = self.frames.setdefault(stamp, {})
        frame[kind] = msg
        while len(self.frames) > 12:
            self.frames.popitem(last=False)
        if not self.required.issubset(frame):
            return
        self.frames.pop(stamp, None)
        try:
            self._process(frame, stamp)
            self.last_frame = time.monotonic()
            self.last_stamp = stamp
        except (ValueError, TypeError, TransformException) as exc:
            self.get_logger().warning(f"RGB-D 帧已丢弃：{exc}")
            self._publish_blocks([], stamp)

    def _process(self, frame, stamp):
        rgb_msg, depth_msg, info = (frame[k] for k in ("rgb", "depth", "info"))
        raw = np.frombuffer(bytes(rgb_msg.data), np.uint8).reshape(rgb_msg.height, rgb_msg.step)
        rgb = raw[:, :rgb_msg.width * 3].reshape(rgb_msg.height, rgb_msg.width, 3)
        if rgb_msg.encoding.lower() == "bgr8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        elif rgb_msg.encoding.lower() != "rgb8":
            raise ValueError(f"unsupported color encoding {rgb_msg.encoding}")
        if depth_msg.encoding.upper() == "32FC1":
            dtype = ">f4" if depth_msg.is_bigendian else "<f4"
            depth = np.frombuffer(bytes(depth_msg.data), dtype=dtype).reshape(
                depth_msg.height, depth_msg.step // 4
            )[:, :depth_msg.width]
        elif depth_msg.encoding.upper() in ("16UC1", "MONO16"):
            dtype = ">u2" if depth_msg.is_bigendian else "<u2"
            depth = np.frombuffer(bytes(depth_msg.data), dtype=dtype).reshape(
                depth_msg.height, depth_msg.step // 2
            )[:, :depth_msg.width].astype(np.float32) * self.depth_scale
            depth[depth <= 0.0] = np.nan
        else:
            raise ValueError(f"unsupported depth encoding {depth_msg.encoding}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError("color and depth are not aligned")
        rotation, translation = self._camera_pose(frame, rgb_msg)
        intrinsic = np.asarray(info.k, dtype=float).reshape(3, 3)
        if self.detector_backend == "yolo":
            detections = detect_yolo_blocks(
                rgb, depth, intrinsic, rotation, translation, self.yolo,
                confidence=self.yolo_confidence, iou=self.yolo_iou,
                device=self.yolo_device, imgsz=self.yolo_imgsz,
                class_map=self.yolo_class_map,
            )
        else:
            detections = detect_blocks(rgb, depth, intrinsic, rotation, translation)
        ball = detect_pingpong_ball(rgb, depth, intrinsic, rotation, translation)
        for detection in detections:
            detection["source"] = self.detection_source
        if isinstance(ball, dict):
            ball["source"] = f"{self.detection_source}_pingpong"
        self._publish_blocks(detections, stamp)
        msg = String()
        msg.data = json.dumps({"ball": ball, "stamp_ns": stamp, "source": self.detection_source})
        self.ball_pub.publish(msg)

    def _camera_pose(self, frame, rgb_msg):
        if self.use_pose_topic:
            pose = frame["pose"]
            if pose.header.frame_id != self.base_frame:
                raise ValueError("camera pose is not expressed in base_frame")
            p, q = pose.pose.position, pose.pose.orientation
        else:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, rgb_msg.header.frame_id,
                Time.from_msg(rgb_msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout),
            ).transform
            p, q = transform.translation, transform.rotation
        rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return rotation, np.array([p.x, p.y, p.z], dtype=float)

    def _publish_blocks(self, detections, stamp):
        msg = String()
        msg.data = json.dumps({
            "count": len(detections), "detections": detections,
            "target": detections[0] if detections else None,
            "source": self.detection_source, "stamp_ns": stamp, "age_sec": 0.0,
        })
        self.block_pub.publish(msg)

    def _expire(self):
        if time.monotonic() - self.last_frame > 1.0:
            self._publish_blocks([], self.last_stamp)


def main(args=None):
    rclpy.init(args=args)
    node = RgbdDetector()
    # 图像回调等待对应时刻的 TF 时，另一线程仍可接收 TF，避免互相阻塞。
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
