"""YOLO 与 D435 深度融合的目标检测函数。

本文件只处理 YOLO 输出、深度采样和坐标变换，不直接依赖 ROS 或
Ultralytics。这样可以在没有安装 Ultralytics 的机器上运行离线测试，
也便于以后替换成其他 YOLO 推理后端。
"""

import json

import numpy as np


def parse_class_map(value):
    """解析“模型类别 -> 抓取配置类别”的 JSON 映射。

    ROS 参数通常以字符串传入，因此这里同时接受 JSON 字符串和字典。
    未列出的类别会保留原名称；映射只用于兼容当前抓取任务已有的
    red/blue/yellow 尺寸和夹爪配置。
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("yolo_class_map 必须是 JSON object（JSON 对象）")
    return {str(key): str(item) for key, item in parsed.items()}


def detect_yolo_blocks(
    rgb,
    depth,
    intrinsic,
    rotation,
    translation,
    predictor,
    *,
    confidence=0.45,
    iou=0.5,
    device="0",
    imgsz=640,
    class_map=None,
):
    """将 YOLO 检测框和同帧 D435 深度转换为抓取目标格式。

    YOLO 只负责识别类别和二维框，目标的三维位置仍由 D435 深度反投影
    得到，再通过同一帧的 TF 变换到 ``base_link``。函数不读取仿真真值。
    ``predictor`` 由节点注入，避免在本文件中强制导入 Ultralytics。
    """
    if rgb.shape[:2] != depth.shape or not np.all(np.isfinite(intrinsic)):
        raise ValueError("RGB/depth 必须对齐，且相机内参必须有限")
    if not callable(predictor):
        raise ValueError("YOLO 推理器不可调用")
    fx, fy, cx, cy = (
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    if fx <= 0 or fy <= 0:
        raise ValueError("焦距内参无效")
    class_map = parse_class_map(class_map)
    kwargs = {
        "conf": float(confidence), "iou": float(iou),
        "imgsz": int(imgsz), "verbose": False,
    }
    if str(device).strip():
        kwargs["device"] = str(device)
    # 本节点统一使用 RGB；Ultralytics 的 NumPy 图像接口要求 BGR。
    # 保留 device 参数，让指定的 GPU 真正参与推理；失败时不丢掉参数重试。
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    outputs = predictor(bgr, **kwargs)
    if outputs is None:
        return []
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    detections = []

    def as_array(value):
        # Ultralytics 可能返回 CUDA Tensor，先搬到主机再用 NumPy 做深度计算。
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    for result in outputs:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = as_array(getattr(boxes, "xyxy", [])).astype(float, copy=False)
        cls = as_array(getattr(boxes, "cls", [])).astype(float, copy=False).reshape(-1)
        conf = as_array(getattr(boxes, "conf", [])).astype(float, copy=False).reshape(-1)
        names = getattr(result, "names", {}) or {}
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            continue
        for index, box in enumerate(xyxy):
            if index >= len(cls) or index >= len(conf):
                continue
            score = float(conf[index])
            if not np.isfinite(score):
                continue
            class_id = int(cls[index])
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            elif 0 <= class_id < len(names):
                class_name = str(names[class_id])
            else:
                class_name = str(class_id)
            x0, y0, x1, y1 = np.clip(
                box,
                [0, 0, 0, 0],
                [rgb.shape[1] - 1, rgb.shape[0] - 1, rgb.shape[1] - 1, rgb.shape[0] - 1],
            )
            if x1 <= x0 or y1 <= y0:
                continue
            # 排除框边缘，避免把桌面或背景深度混入物体距离。
            u0 = int(np.floor(x0 + 0.2 * (x1 - x0)))
            u1 = int(np.ceil(x1 - 0.2 * (x1 - x0)))
            v0 = int(np.floor(y0 + 0.2 * (y1 - y0)))
            v1 = int(np.ceil(y1 - 0.2 * (y1 - y0)))
            sample = np.asarray(depth[v0 : v1 + 1, u0 : u1 + 1], dtype=float)
            valid = sample[np.isfinite(sample) & (sample > 0.05) & (sample < 3.0)]
            if valid.size < 4:
                continue
            z = float(np.median(valid))
            u, v = float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)
            optical = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
            position = optical @ np.asarray(rotation, dtype=float).T + translation
            width_m = float((x1 - x0) * z / fx)
            height_m = float((y1 - y0) * z / fy)
            grasp_color = class_map.get(class_name, class_name)
            detections.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": score,
                    "color": grasp_color,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "width_m": width_m,
                    "height_m": height_m,
                    "longest_m": max(width_m, height_m),
                    "shortest_m": min(width_m, height_m),
                    "grasp_yaw_rad": 0.0,
                    "pixels": int(max(0, round((x1 - x0) * (y1 - y0)))),
                    "source": "yolo_rgbd",
                }
            )
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


__all__ = ["detect_yolo_blocks", "parse_class_map"]
