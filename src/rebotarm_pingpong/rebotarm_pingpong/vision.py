"""基于双目 RGB 图像的乒乓球定位，与仿真球状态完全隔离。

观测器的检测和跟踪只接收渲染或真实相机图像、固定标定参数和采集时间戳：
先检测白球，再用双目射线三角测量位置，最后结合时间戳估计速度和传输延迟。
代码不会直接读取深度缓冲、分割图或球体位姿；视觉失败时也不会回退到仿真真值。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import mujoco
import numpy as np


@dataclass(frozen=True)
class PixelDetection:
    """单个相机看到的球心、像素半径和检测置信度。"""

    center: np.ndarray
    radius_px: float
    confidence: float


@dataclass(frozen=True)
class CameraCalibration:
    """针孔相机标定参数；旋转矩阵把相机坐标轴转换到世界坐标。"""

    name: str
    position: np.ndarray
    rotation: np.ndarray
    width: int
    height: int
    focal_px: float

    @property
    def principal_point(self):
        return np.array([(self.width - 1) / 2., (self.height - 1) / 2.])

    def ray(self, pixel):
        u, v = (np.asarray(pixel) - self.principal_point) / self.focal_px
        direction = self.rotation @ np.array([u, -v, -1.])
        return direction / np.linalg.norm(direction)

    def project(self, position):
        local = self.rotation.T @ (np.asarray(position) - self.position)
        depth = -local[2]
        if depth <= 0:
            return np.full(2, np.nan)
        return self.principal_point + self.focal_px * local[:2] / [depth, -depth]

    @classmethod
    def from_look_at(cls, name, position, look_at, width=320, height=240, fovy_deg=48.):
        """按相机位置和观察点生成针孔标定，可用于仿真相机或真机适配器。"""
        position = np.asarray(position, dtype=float)
        z = position - np.asarray(look_at, dtype=float)
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        if np.linalg.norm(x) < 1e-9:
            x = np.cross([0., 1., 0.], z)
        x /= np.linalg.norm(x)
        return cls(name, position, np.column_stack([x, np.cross(z, x), z]),
                   width, height, height / (2 * np.tan(np.deg2rad(fovy_deg) / 2)))


@dataclass(frozen=True)
class BallEstimate:
    """由视觉得到的球状态；``valid`` 表示当前估计是否足够新且有速度。"""

    position: np.ndarray
    velocity: np.ndarray
    valid: bool
    age_s: float
    confidence: float
    measurement_timestamp_s: float
    received_timestamp_s: float
    detections: tuple[PixelDetection | None, PixelDetection | None]
    velocity_valid: bool = False

    @property
    def latency_s(self):
        return max(0., self.received_timestamp_s - self.measurement_timestamp_s)


def detect_white_ball(image, settings=None):
    """检测图像中的白色圆形目标，立体几何再负责排除误检。

    这是外观检测，不读取仿真对象 ID；白色圆物体可能造成误检，完全遮挡时返回空列表。
    """
    s = settings or {}
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("ball detector requires an HxWx3 RGB image")
    # 使用 OpenCV 的 uint8 归约完成整幅阈值处理，避免 NumPy min/max 产生多份
    # 浮点临时数组。通道范围不超过 255，因此 uint8 相减不会溢出。
    channels = cv2.split(rgb)
    brightness = cv2.min(cv2.min(channels[0], channels[1]), channels[2])
    chroma = cv2.max(cv2.max(channels[0], channels[1]), channels[2]) - brightness
    mask = cv2.inRange(brightness, float(s.get("min_brightness", 100)), 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(chroma, 0, float(s.get("max_chroma", 60))))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    detections = []
    for contour in contours:
        if len(contour) < 8:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not float(s.get("min_radius_px", 2.)) <= radius <= float(s.get("max_radius_px", 25.)):
            continue
        # 灯光下的空心球下半部可能较暗。先以白色区域为种子，再纳入高于背景
        # 的局部轮廓，避免只按白色阈值时把球心估计得过高。
        margin = int(np.ceil(1.5 * radius))
        left, top = max(0, int(x) - margin), max(0, int(y) - margin)
        right, bottom = min(rgb.shape[1], int(x) + margin + 1), min(rgb.shape[0], int(y) + margin + 1)
        patch_brightness = brightness[top:bottom, left:right]
        yy, xx = np.indices(patch_brightness.shape)
        distance = np.hypot(xx + left - x, yy + top - y)
        ring = patch_brightness[distance > 1.2 * radius]
        background = float(np.median(ring)) if ring.size else 0.
        silhouette_threshold = max(float(s.get("min_brightness", 100)) * .45, background + 20.)
        local_mask = ((patch_brightness >= silhouette_threshold) &
                      (chroma[top:bottom, left:right] <= float(s.get("max_chroma", 60)))).astype(np.uint8)
        local_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for local_contour in local_contours:
            if cv2.pointPolygonTest(local_contour, (x - left, y - top), False) < 0:
                continue
            (local_x, local_y), local_radius = cv2.minEnclosingCircle(local_contour)
            if radius <= local_radius <= 1.4 * radius:
                contour = local_contour + np.array([[[left, top]]], dtype=np.int32)
                x, y, radius = local_x + left, local_y + top, local_radius
            break
        perimeter, area = cv2.arcLength(contour, True), cv2.contourArea(contour)
        circularity = 4 * np.pi * area / max(perimeter ** 2, 1e-12)
        fill = area / max(np.pi * radius ** 2, 1e-12)
        if circularity < .60 or fill < .55:
            continue
        # 外接圆能包含阴影下边界；直接用图像矩会偏向受光半球，导致球心偏移。
        confidence = float(np.clip(np.sqrt(circularity * fill), 0., 1.))
        detections.append(PixelDetection(np.array([x, y]), float(radius), confidence))
    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def triangulate(first, second, first_pixel, second_pixel):
    """求两条标定射线的最小二乘交点，返回位置和两条射线的间隙。"""
    ray1, ray2 = first.ray(first_pixel), second.ray(second_pixel)
    if abs(np.dot(ray1, ray2)) > .995:
        return None, np.inf
    depths = np.linalg.lstsq(np.column_stack([ray1, -ray2]),
                            second.position - first.position, rcond=None)[0]
    if np.any(depths <= 0):
        return None, np.inf
    point1 = first.position + depths[0] * ray1
    point2 = second.position + depths[1] * ray2
    return (point1 + point2) / 2., float(np.linalg.norm(point1 - point2))


class StereoBallTracker:
    """只依赖图像的双目定位器，并按时间戳估计飞行中的位置和速度。"""

    def __init__(self, calibrations, settings=None, gravity=9.81):
        if len(calibrations) != 2:
            raise ValueError("ball localization requires two calibrated RGB cameras")
        self.cameras = tuple(calibrations)
        self.settings = settings or {}
        self.gravity = np.array([0., 0., -float(gravity)])
        self.reset()

    def reset(self):
        """清空历史测量；下一帧只提供位置，第二帧后才启用速度。"""

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_measured_position = None
        self.capture_time = self.received_time = -1.
        self.measurements = 0
        self.quality = 0.
        self.detections = (None, None)

    def locate(self, images, rng=None):
        """在两路图像中配对候选球，并用重投影和工作空间约束筛选。"""

        candidates = [detect_white_ball(image, self.settings) for image in images]
        best = None
        min_corner = np.asarray(self.settings.get("workspace_min_m", [-2., -2., -1.]))
        max_corner = np.asarray(self.settings.get("workspace_max_m", [2., 2., 2.]))
        for first in candidates[0][:8]:
            for second in candidates[1][:8]:
                pixels = [first.center.copy(), second.center.copy()]
                jitter = float(self.settings.get("centroid_noise_std_px", .15))
                if rng is not None and jitter:
                    pixels = [pixel + rng.normal(0., jitter, 2) for pixel in pixels]
                position, gap = triangulate(*self.cameras, *pixels)
                if position is None or gap > float(self.settings.get("max_ray_gap_m", .012)):
                    continue
                if np.any(position < min_corner) or np.any(position > max_corner):
                    continue
                reprojection = max(np.linalg.norm(camera.project(position) - pixel)
                                   for camera, pixel in zip(self.cameras, pixels))
                if reprojection > float(self.settings.get("max_reprojection_error_px", 2.)):
                    continue
                radius_m = float(self.settings.get("ball_radius_m", .020))
                apparent_radii = [radius_m * camera.focal_px /
                                  np.linalg.norm(position - camera.position) for camera in self.cameras]
                if any(not .55 < detection.radius_px / radius < 1.6
                       for detection, radius in zip([first, second], apparent_radii)):
                    continue
                confidence = min(first.confidence, second.confidence) * np.exp(-reprojection / 2.)
                if best is None or confidence > best[1]:
                    best = position, float(confidence), (first, second)
        if best is None:
            return None, 0., (None, None)
        return best

    def accept(self, position, confidence, detections, capture_time, received_time):
        """接收带采集/到达时间戳的测量；过期或无效帧不会覆盖当前状态。"""

        self.detections = detections
        if position is None or capture_time <= self.capture_time:
            return
        position = np.asarray(position, dtype=float)
        if not np.isfinite(position).all():
            return
        if self.measurements and capture_time - self.capture_time > float(self.settings.get("max_age_s", .10)):
            # 长时间遮挡期间可能发生未知碰撞，重新看到的一个位置无法恢复速度；
            # 因此重新获取后要再等两帧新数据。
            self.measurements = 0
        if self.measurements:
            dt = float(capture_time - self.capture_time)
            # 有限差分对应两帧中点速度，再用重力推进到当前采样时刻。这里不读取
            # 接触真值，碰撞只能由后续相机帧中的速度变化推断。
            measured_velocity = (position - self.last_measured_position) / dt + .5 * self.gravity * dt
            predicted_velocity = self.velocity + self.gravity * dt
            predicted_position = self.position + self.velocity * dt + .5 * self.gravity * dt ** 2
            velocity_alpha = float(self.settings.get("velocity_alpha", .80))
            self.velocity = velocity_alpha * measured_velocity + (1 - velocity_alpha) * predicted_velocity
            position_alpha = float(self.settings.get("position_alpha", .85))
            self.position = position_alpha * position + (1 - position_alpha) * predicted_position
        else:
            self.position = position.copy()
            self.velocity = np.zeros(3)
        self.last_measured_position = position.copy()
        self.capture_time, self.received_time = float(capture_time), float(received_time)
        self.measurements += 1
        self.quality = float(confidence)

    def process(self, images, capture_time, received_time, rng=None):
        """处理同步、去畸变的 RGB 帧；输入可以来自仿真渲染器或真实相机。"""

        self.accept(*self.locate(images, rng), capture_time, received_time)
        return self.estimate(received_time)

    def estimate(self, now):
        """把最近一次测量按重力外推到当前时刻，并标记估计是否仍然有效。"""

        max_age = float(self.settings.get("max_age_s", .10))
        age = max(0., float(now) - self.capture_time) if self.measurements else max_age
        dt = min(age, max_age) if self.measurements else 0.
        position = self.position + self.velocity * dt + .5 * self.gravity * dt ** 2
        velocity = self.velocity + self.gravity * dt
        velocity_valid = self.measurements >= 2
        valid = velocity_valid and age <= max_age
        confidence = self.quality * np.exp(-age / max_age) if valid else 0.
        return BallEstimate(position.copy(), velocity.copy(), valid, float(age), float(confidence),
                            self.capture_time, self.received_time, self.detections, velocity_valid)


class CameraBallObserver:
    """调度 RGB 采集和传输延迟，并只输出视觉估计状态。

    相机位姿是从模型读取的一次性世界标定。``data`` 只用于渲染图像和读取时钟；
    本类从不读取仿真球的位置、速度、深度或分割掩码，因此视觉失败不会真值回退。
    """

    def __init__(self, model, camera_config, perception_config=None, gravity=9.81,
                 renderer_factory: Callable | None = None):
        self.model = model
        self.settings = perception_config or {}
        self.camera_config = camera_config
        self.renderer_factory = renderer_factory or mujoco.Renderer
        self.width, self.height = int(self.settings.get("width", 320)), int(self.settings.get("height", 240))
        names = self.settings.get("camera_names", ["tracking", "tracking_side"])
        if len(names) != 2:
            raise ValueError("perception.camera_names must contain two cameras")
        self.camera_names = [name if name.startswith("pingpong_") else f"pingpong_{name}" for name in names]
        calibrations = []
        for name in self.camera_names:
            try:
                camera_id = model.camera(name).id
            except KeyError as exc:
                raise ValueError(f"missing calibrated RGB camera {name}") from exc
            if model.cam_bodyid[camera_id] != 0:
                raise ValueError("ball tracking cameras must be fixed to the world")
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.cam_quat[camera_id])
            focal = self.height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
            calibrations.append(CameraCalibration(name, model.cam_pos[camera_id].copy(),
                                                  rotation.reshape(3, 3), self.width, self.height, focal))
        self.tracker = StereoBallTracker(calibrations, self.settings, gravity)
        self.nominal_calibrations = tuple(calibrations)
        self.renderer = None
        self.reset(np.random.default_rng())

    def reset(self, rng=None):
        """重置跟踪器和传感器队列，并为本回合抽取固定的标定误差。"""

        self.rng = rng if rng is not None else np.random.default_rng()
        # 真机的内外参存在误差；每回合抽取一次固定偏差。渲染相机姿态保持不变，
        # 该随机化只作用于观测器标定，不涉及球的状态。
        position_std = float(self.settings.get("calibration_position_std_m", 0.))
        angle_std = np.deg2rad(float(self.settings.get("calibration_rotation_std_deg", 0.)))
        focal_std = float(self.settings.get("calibration_focal_relative_std", 0.))
        calibrations = []
        for camera in self.nominal_calibrations:
            offset = self.rng.normal(0., position_std, 3) if position_std else np.zeros(3)
            rotation_error = cv2.Rodrigues(self.rng.normal(0., angle_std, 3))[0] if angle_std else np.eye(3)
            focal_scale = 1 + self.rng.normal(0., focal_std) if focal_std else 1.
            calibrations.append(CameraCalibration(camera.name, camera.position + offset,
                                                  rotation_error @ camera.rotation,
                                                  camera.width, camera.height,
                                                  camera.focal_px * focal_scale))
        self.tracker.cameras = tuple(calibrations)
        self.tracker.reset()
        self.pending = deque()
        self.exposure_samples = []
        self.next_capture = 0.
        self.frames_captured = self.frames_missed = 0
        self.estimate = self.tracker.estimate(0.)

    def _render_images(self, data):
        """分别渲染两台固定相机的 RGB 图像；球只通过图像进入检测器。"""

        if self.renderer is None:
            self.renderer = self.renderer_factory(self.model, self.height, self.width)
        images = []
        for camera in self.camera_names:
            self.renderer.update_scene(data, camera=camera)
            images.append(self.renderer.render().copy())
        return images

    def _sensor_images(self, images):
        """模拟曝光后的模糊和像素噪声，得到传感器实际交给检测器的图像。"""

        result = []
        for image in images:
            blur = float(self.settings.get("blur_sigma_px", .35))
            if blur:
                image = cv2.GaussianBlur(image, (3, 3), blur)
            noise = float(self.settings.get("image_noise_std", 1.5))
            if noise:
                # 不为每个像素创建 float64 临时数组，降低查看器延迟。对于 8 位
                # 传感器模型，float32 已足够；同一个回合随机种子仍保持确定性。
                image = np.asarray(image, dtype=np.float32)
                image += self.rng.standard_normal(image.shape, dtype=np.float32) * np.float32(noise)
                image = np.clip(image, 0., 255.).astype(np.uint8)
            result.append(image)
        return result

    def observe(self, data):
        """按相机帧率采集、加入丢帧/延迟，再交给双目跟踪器。"""

        now = float(data.time)
        exposure = float(self.settings.get("exposure_s", .002))
        if exposure and self.next_capture - exposure <= now < self.next_capture - 1e-10:
            # 在曝光时间内积累真实渲染子帧；相同时间戳的重复调用不会重复采样。
            # 不读取未来状态，也不插值球的真值位姿。
            if not self.exposure_samples or now > self.exposure_samples[-1][0] + 1e-10:
                self.exposure_samples.append((now, self._render_images(data)))
        if now + 1e-10 >= self.next_capture:
            period = 1. / float(self.settings.get("fps", 60.))
            self.next_capture += (int(max(0., now - self.next_capture) / period) + 1) * period
            self.frames_captured += 1
            capture_time = now
            if self.rng.random() < float(self.settings.get("dropout_probability", .01)):
                observation = None, 0., (None, None)
            else:
                self.exposure_samples.append((now, self._render_images(data)))
                capture_time = float(np.mean([sample[0] for sample in self.exposure_samples]))
                images = [np.mean([sample[1][index] for sample in self.exposure_samples], axis=0).astype(np.uint8)
                          for index in range(2)]
                observation = self.tracker.locate(self._sensor_images(images), self.rng)
            self.exposure_samples = []
            if observation[0] is None:
                self.frames_missed += 1
            latency = max(0., float(self.settings.get("latency_s", .012)) +
                          self.rng.normal(0., float(self.settings.get("latency_jitter_s", .002))))
            self.pending.append((now + latency, capture_time, observation))
        # 延迟抖动可能让帧乱序到达；tracker.accept 会拒绝早于当前状态的迟到帧。
        ready = sorted((packet for packet in self.pending if packet[0] <= now + 1e-10), key=lambda packet: packet[0])
        self.pending = deque(packet for packet in self.pending if packet[0] > now + 1e-10)
        for _, capture_time, observation in ready:
            self.tracker.accept(*observation, capture_time, now)
        self.estimate = self.tracker.estimate(now)
        return self.estimate

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
