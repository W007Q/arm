"""Example upstream MIT publisher: capture measured positions, stream a hold.

Does not call enable. tau_ff remains the explicitly supplied vector; this is
not a gravity model and must not be mistaken for gravity compensation.
"""
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rebotarm_msgs.msg import JointMotorState
from std_msgs.msg import Float64MultiArray


class HoldPublisher(Node):
    def __init__(self):
        super().__init__('uart_mit_hold_example')
        self.declare_parameter('arm_namespace','rebotarm')
        self.declare_parameter('joint_names',[f'joint{i}' for i in range(1,7)]+['gripper'])
        self.declare_parameter('kp',10.)
        self.declare_parameter('kd',1.)
        self.declare_parameter('tau_ff',[0.]*7)
        self.declare_parameter('rate',50.)
        get=lambda key:self.get_parameter(key).value
        self.names=list(get('joint_names'));self.kp=float(get('kp'));self.kd=float(get('kd'))
        self.tau=list(get('tau_ff'));rate=float(get('rate'))
        if len(self.tau)!=len(self.names) or not all(math.isfinite(x) for x in self.tau+[self.kp,self.kd,rate]) or rate<=0:
            raise ValueError('finite gains, rate and one tau_ff per joint required')
        namespace=str(get('arm_namespace')).strip('/')
        prefix=f'/{namespace}' if namespace else ''
        self.pub=self.create_publisher(Float64MultiArray,prefix+'/uart/mit_targets',1)
        self.create_subscription(JointState,prefix+'/joint_states',self.receive,10)
        if 'gripper' in self.names:
            self.create_subscription(JointMotorState,prefix+'/gripper/state',
                                     lambda msg:self.capture({'gripper':msg.position}),10)
        self.target=None;self.measured={};self.feedback_stamps={}
        self.create_timer(1./rate,self.publish)
        self.get_logger().info('Waiting for measured joints; will stream hold targets without enabling motors.')

    def receive(self,msg):
        self.capture(dict(zip(msg.name,msg.position)))

    def capture(self,values):
        now=time.monotonic()
        for name,value in values.items():
            if name in self.names and math.isfinite(value):
                self.measured[name]=value;self.feedback_stamps[name]=now
        if self.target is None and all(now-self.feedback_stamps.get(name,0.)<.2 for name in self.names):
            self.target=[self.measured[name] for name in self.names]
            self.get_logger().info(f'Captured hold positions: {self.target}; enable is a separate service.')

    def publish(self):
        if self.target is None or any(time.monotonic()-self.feedback_stamps.get(name,0.)>.2 for name in self.names):return
        msg=Float64MultiArray()
        msg.data=[value for q,tau in zip(self.target,self.tau) for value in (q,0.,self.kp,self.kd,tau)]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args);node=HoldPublisher()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
