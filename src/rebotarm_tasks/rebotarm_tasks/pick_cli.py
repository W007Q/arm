from __future__ import annotations

import argparse
import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros_task_client import RosTaskClient


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="One-shot: detect a color block and grasp it."
    )
    parser.add_argument("--namespace", default="rebotarm")
    parser.add_argument("--config", default="", help="pick_<backend>.yaml task profile")
    parser.add_argument(
        "--color",
        default="auto",
        choices=("auto", "red", "blue", "yellow"),
    )
    parser.add_argument("--approach-z", type=float, default=0.18)
    parser.add_argument("--grasp-z", type=float, default=0.140)
    parser.add_argument("--lift-z", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--service-timeout-sec", type=float, default=2.0)
    parser.add_argument("--action-timeout-padding-sec", type=float, default=8.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=15.0)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, ros_args = _parse_args(argv)

    rclpy.init(args=ros_args)
    bridge = RosTaskClient(
        namespace=args.namespace,
        config_path=args.config,
        motion_allowed=True,
        service_timeout_sec=args.service_timeout_sec,
        action_timeout_padding_sec=args.action_timeout_padding_sec,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        deadline = time.monotonic() + max(float(args.ready_timeout_sec), 0.0)
        while time.monotonic() < deadline:
            topics = bridge.get_status()["topics"]
            if (
                topics.get("joint_states_age_sec") is not None
                and topics.get("vision_age_sec") is not None
            ):
                break
            time.sleep(0.1)

        result = bridge.pick_color(
            args.color,
            args.approach_z,
            args.grasp_z,
            args.lift_z,
            args.duration,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
