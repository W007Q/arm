import numpy as np
import pytest

from rebotarm_vision.yolo_detection import detect_yolo_blocks, parse_class_map


class _Boxes:
    xyxy = np.array([[10.0, 10.0, 30.0, 30.0]])
    cls = np.array([0.0])
    conf = np.array([0.91])


class _Result:
    boxes = _Boxes()
    names = {0: "red_cube"}


def test_yolo_box_is_fused_with_depth_and_camera_pose():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb[:] = [230, 80, 10]
    depth = np.full((50, 50), 0.5, dtype=np.float32)
    intrinsic = np.array([[100.0, 0.0, 25.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    translation = np.array([0.1, 0.2, 0.3])
    calls = []

    def predictor(image, **kwargs):
        calls.append((image, kwargs))
        return [_Result()]

    detections = detect_yolo_blocks(
        rgb, depth, intrinsic, rotation, translation, predictor,
        device="0", imgsz=960, class_map={"red_cube": "red"},
    )
    assert len(detections) == 1
    target = detections[0]
    assert target["class_name"] == "red_cube"
    assert target["color"] == "red"
    assert target["source"] == "yolo_rgbd"
    assert target["x"] == pytest.approx(0.075)
    assert target["y"] == pytest.approx(0.175)
    assert target["z"] == pytest.approx(0.8)
    assert calls[0][1]["device"] == "0"
    assert calls[0][1]["imgsz"] == 960
    # RGB 输入必须转成 Ultralytics 要求的 BGR，且不能修改原始相机图像。
    assert calls[0][0][0, 0].tolist() == [10, 80, 230]
    assert calls[0][0].flags.c_contiguous
    assert rgb[0, 0].tolist() == [230, 80, 10]


def test_gpu_inference_error_does_not_retry_without_device():
    calls = []

    def predictor(image, **kwargs):
        calls.append(kwargs)
        raise TypeError("推理失败")

    with pytest.raises(TypeError, match="推理失败"):
        detect_yolo_blocks(
            np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10)),
            np.eye(3), np.eye(3), np.zeros(3), predictor,
        )
    assert len(calls) == 1
    assert calls[0]["device"] == "0"


def test_yolo_invalid_depth_is_rejected_without_fake_position():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    depth = np.zeros((50, 50), dtype=np.float32)
    intrinsic = np.eye(3)
    result = detect_yolo_blocks(
        rgb, depth, intrinsic, np.eye(3), np.zeros(3), lambda *_a, **_k: [_Result()]
    )
    assert result == []


def test_class_map_requires_a_json_object():
    assert parse_class_map('{"cup":"red"}') == {"cup": "red"}
    try:
        parse_class_map("[]")
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("invalid class map was accepted")
