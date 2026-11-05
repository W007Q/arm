"""Legacy forwarding path for external clients; new code imports rebotarm_vision.rgbd_detector."""
from rebotarm_vision.rgbd_detector import RgbdDetector, main

RsSceneDetector = RgbdDetector

__all__ = ["RgbdDetector", "RsSceneDetector", "main"]
