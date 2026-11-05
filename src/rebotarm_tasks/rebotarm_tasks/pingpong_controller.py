"""Legacy ROS sinusoidal motion demo retained for existing callers.

Ball observations are received but do not affect control. For the independent
physical paddle scene, closed-loop vertical baseline and residual PPO, use
rebotarm_pingpong.cli; see docs/pingpong_zh.md. This demo is not that controller.
"""
import argparse
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rebotarm_msgs.msg import JointPosVelCmd
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class PingPongController(Node):
    def __init__(self, namespace: str, amplitude: float, frequency: float):
        super().__init__("rebotarm_pingpong_controller")
        ns = namespace.strip("/")
        self.amp = max(0.0, float(amplitude))
        self.freq = max(0.05, float(frequency))
        self.q = None
        self.q0 = None
        self.ball = None
        self.started = self.get_clock().now()
        self.create_subscription(
            JointState, f"/{ns}/joint_states", self._joints, qos_profile_sensor_data
        )
        self.create_subscription(String, f"/{ns}/vision/pingpong_ball", self._ball, 10)
        self.pub = [
            self.create_publisher(JointPosVelCmd, f"/{ns}/joints/joint{i}/cmd/pos_vel", 10)
            for i in range(1, 7)
        ]
        self.create_timer(0.004, self._tick)

    def _joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        names = [f"joint{i}" for i in range(1, 7)]
        if all(name in values for name in names):
            self.q = [float(values[name]) for name in names]
            if self.q0 is None:
                self.q0 = list(self.q)

    def _ball(self, msg):
        try:
            packet = json.loads(msg.data)
            self.ball = packet.get("ball")
        except (TypeError, ValueError):
            self.ball = None

    def _tick(self):
        if self.q is None:
            return
        elapsed = (self.get_clock().now() - self.started).nanoseconds * 1e-9
        target = list(self.q0)
        # Conservative vertical stroke baseline.  The D435 predictor replaces
        # this phase term after the contact model and calibration are validated.
        target[1] += self.amp * math.sin(2.0 * math.pi * self.freq * elapsed)
        for index, pub in enumerate(self.pub):
            msg = JointPosVelCmd()
            msg.pos = target[index]
            msg.vlim = 0.35
            msg.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="rebotarm")
    parser.add_argument("--amplitude-rad", type=float, default=0.08)
    parser.add_argument("--frequency-hz", type=float, default=1.2)
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = PingPongController(parsed.namespace, parsed.amplitude_rad, parsed.frequency_hz)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
