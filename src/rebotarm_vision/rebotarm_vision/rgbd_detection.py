"""RGB-D 颜色分割和乒乓球检测。

本文件保留颜色识别作为备用后端。正式抓取默认使用
``rebotarm_vision.yolo_detection`` 中的 YOLO 后端。
"""
import cv2
import numpy as np
from .yolo_detection import detect_yolo_blocks, parse_class_map


def detect_blocks(rgb, depth, intrinsic, rotation, translation, min_area=25):
    if rgb.shape[:2] != depth.shape or not np.all(np.isfinite(intrinsic)):
        raise ValueError('RGB/depth 必须对齐，且相机内参必须有限')
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError('焦距内参无效')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hues = {'red': [(0, 10), (170, 179)], 'blue': [(95, 135)], 'yellow': [(18, 38)]}
    detections = []
    for color, intervals in hues.items():
        mask = np.zeros(depth.shape, np.uint8)
        for lo, hi in intervals:
            mask |= cv2.inRange(hsv, (lo, 100, 65), (hi, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < min_area:
                continue
            v, u = np.nonzero(labels == index)
            z = depth[v, u]
            valid = np.isfinite(z) & (z > 0.02) & (z < 3.)
            if valid.sum() < min_area:
                continue
            v, u, z = v[valid], u[valid], z[valid]
            optical = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
            world = optical @ rotation.T + translation
            # 只使用可见上表面；不从仿真器读取隐藏尺寸。
            top = float(np.percentile(world[:, 2], 85))
            surface = world[np.abs(world[:, 2] - top) <= .005]
            if len(surface) < 10:
                continue
            rect = cv2.minAreaRect(surface[:, :2].astype(np.float32))
            corners = cv2.boxPoints(rect)
            edges = np.roll(corners, -1, axis=0) - corners
            lengths = np.linalg.norm(edges, axis=1)
            edge = edges[int(np.argmax(lengths))]
            longest, shortest = float(max(lengths)), float(min(lengths))
            if shortest < .008 or longest > .15:
                continue
            x, y = rect[0]
            detections.append(dict(color=color, x=float(x), y=float(y), z=top,
                                   surface_z=top, width_m=shortest, longest_m=longest,
                                   shortest_m=shortest, grasp_yaw_rad=(0.0 if longest/shortest < 1.15 else float(np.arctan2(edge[1], edge[0]))),
                                   source='camera_rgbd', pixels=int(len(surface))))
    return detections


def detect_pingpong_ball(rgb, depth, intrinsic, rotation, translation):
    """从对齐的 D435 RGB-D 帧中检测白色乒乓球。"""
    if rgb.shape[:2] != depth.shape:
        raise ValueError('RGB/depth 必须对齐')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 145), (179, 75, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 4 or area > 500 or min(w, h) <= 0 or max(w, h) / min(w, h) > 2.2:
            continue
        v, u = np.nonzero(labels == i)
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 0.05) & (z < 3.0)
        if valid.sum() < 4:
            continue
        z = z[valid]; u = u[valid]; v = v[valid]
        optical = np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))
        world = optical @ rotation.T + translation
        candidates.append((area, world.mean(axis=0)))
    if not candidates:
        return None
    area, position = max(candidates, key=lambda item: item[0])
    return {'position_m': [float(x) for x in position], 'pixels': area,
            'source': 'camera_rgbd_pingpong'}


__all__ = [
    "detect_blocks",
    "detect_pingpong_ball",
    # 兼容旧调用方；新代码请从 yolo_detection 导入。
    "detect_yolo_blocks",
    "parse_class_map",
]
