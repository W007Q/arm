"""Legacy forwarding path for external clients; new code imports rebotarm_tasks.pingpong_controller."""
from rebotarm_tasks.pingpong_controller import PingPongController, main

__all__ = ["PingPongController", "main"]
