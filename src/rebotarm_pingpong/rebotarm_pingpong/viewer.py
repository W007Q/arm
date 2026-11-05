"""交互式查看器的相机设置，与仿真中的传感器相机相互独立。

查看器的自由相机只影响人看到的画面；选择 `tracking` 或 `tracking_side`
时才会切换到模型中定义的固定观察位姿，不会改变控制器使用的 RGB 图像。
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import mujoco
import numpy as np


def configure_camera(camera, model, config, mode="free"):
    """设置查看器相机模式；调用方应在持有 viewer 锁时调用本函数。

    `free`/`overview` 使用配置中的总览位置，允许鼠标拖动和缩放；
    `tracking`/`tracking_side` 选择模型里的固定相机，方便预览传感器视角。
    """
    if mode not in ("free", "overview", "tracking", "tracking_side"):
        raise ValueError(f"unknown viewer camera: {mode}")
    mujoco.mjv_defaultCamera(camera)
    if mode != "free":
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = model.camera(f"pingpong_{mode}").id
        return
    overview = config["cameras"]["overview"]
    target = np.array(overview["look_at_m"], dtype=float)
    offset = np.array(overview["position_m"], dtype=float) - target
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = float(np.linalg.norm(offset))
    # MuJoCo 的方位角表示观察方向，不是相机眼睛所在位置。
    camera.azimuth = math.degrees(math.atan2(-offset[1], -offset[0]))
    camera.elevation = -math.degrees(math.atan2(offset[2], math.hypot(offset[0], offset[1])))


def close_camera_window(viewer, timeout_s=3.):
    """等待 GLFW 渲染线程完成清理后再退出解释器。

    MuJoCo 的被动窗口 `close()` 只请求退出，不会等待渲染线程；轮询 `viewer.m`
    可以覆盖原生资源尚未释放的短暂阶段，避免程序退出时出现窗口或线程错误。
    """
    viewer.close()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if viewer.m is None:
                return
        except mujoco.UnexpectedError:
            # 原生资源仍在清理，继续等待句柄失效。
            pass
        time.sleep(.01)
    raise RuntimeError("MuJoCo camera window did not finish closing")


@contextmanager
def camera_window(model, data):
    """以上下文管理器方式打开被动查看器，并在离开时可靠关闭。"""
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        yield viewer
    finally:
        close_camera_window(viewer)
