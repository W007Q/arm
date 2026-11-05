"""Publish identical gripper state fields for MuJoCo, CAN and UART."""
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rebotarm_msgs.msg import JointMotorState
from std_msgs.msg import String
from .settings import load_settings
from .gripper import MujocoGripperAdapter, Pa43GripperAdapter

class GripperStateNode(Node):
    def __init__(self):
        super().__init__('rebotarm_gripper_state')
        self.declare_parameter('arm_namespace', 'rebotarm')
        self.declare_parameter('config_path', '')
        self.settings = load_settings(self.get_parameter('config_path').value)
        ns = self.get_parameter('arm_namespace').value.strip('/')
        self.pub = self.create_publisher(String, f'/{ns}/gripper/status', 10)
        if self.settings.backend == 'mujoco':
            self.adapter = MujocoGripperAdapter(self.settings)
            self.create_subscription(JointState, f'/{ns}/joint_states', self.sim_state, qos_profile_sensor_data)
        else:
            self.adapter = Pa43GripperAdapter(self.settings, self.settings.backend)
            self.create_subscription(JointMotorState, f'/{ns}/gripper/state', self.motor_state, qos_profile_sensor_data)
    def publish(self, state):
        msg = String(); msg.data = json.dumps(state.payload()); self.pub.publish(msg)
    def sim_state(self, msg):
        try:
            self.publish(self.adapter.convert(dict(zip(msg.name,msg.position)), dict(zip(msg.name,msg.velocity)), dict(zip(msg.name,msg.effort))))
        except (KeyError, ValueError):
            pass
    def motor_state(self, msg):
        self.publish(self.adapter.convert(msg.position,msg.velocity,msg.torque,msg.status_code))

def main(args=None):
    rclpy.init(args=args); node = GripperStateNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
