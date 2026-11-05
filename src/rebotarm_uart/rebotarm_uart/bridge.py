"""ROS 与 UART 状态机之间的适配层。

发送路径是 ``ROS topic -> _single/_batch -> link.set_targets ->
link._tick -> protocol.pack_command -> PosixUart.write``；接收路径是
``PosixUart.read -> protocol.Decoder -> link._receive ->
_publish_feedback/_publish_status -> ROS``。本文件只做消息映射和参数
装配，不应在回调里直接拼 UART 字节。STM32 收到命令后负责 CAN 下发和
电机反馈汇总。
"""
import json
import math

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rebotarm_msgs.msg import JointMitCmd, JointMotorState, ArmStatus
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from .link import UartLink, Limits
from . import protocol as protocol
from .protocol import MitTarget
from .transport import PosixUart
from rebotarm_description.parameters import uart_parameters


def _planning_position(value, limit, zero_tolerance=0.):
    """将已确认的零位小误差与编码器边界量化误差对齐给模型。

    零位死区由硬件配置逐轴开启；仅影响模型 joint_states，原始反馈仍保留。
    CAN 反馈常把零位返回为 -0.000191 rad，而 J3 的模型下限是 0。
    只有四舍五入到 0.001 rad 后落在合法边界内时才对齐；明显越限的
    反馈保持原值，避免掩盖真实的零位或方向错误。
    """
    value = float(value)
    if zero_tolerance > 0 and abs(value) <= zero_tolerance and limit.lower <= 0 <= limit.upper:
        return 0.
    encoded = round(value * 1000.0) / 1000.0
    if value < limit.lower and encoded >= limit.lower:
        return float(limit.lower)
    if value > limit.upper and encoded <= limit.upper:
        return float(limit.upper)
    return value


class UartBridge(Node):
    """把 ROS 的 MIT 目标和实测电机状态接到 UartLink。"""

    def __init__(self):
        super().__init__('rebotarm_uart_bridge')
        # 关节限位、ID、增益和串口安全策略来自统一硬件配置；这里不复制
        # 一份容易漂移的手工表，uart.yaml 由同步脚本派生。
        defaults = uart_parameters()
        for name,value in defaults.items():self.declare_parameter(name,value)
        get=lambda name:self.get_parameter(name).value
        port=str(get('port'))
        if not port:
            raise ValueError('port must explicitly name the STM32 UART, e.g. /dev/serial/by-id/...')
        self.names=list(get('joint_names'));self.ids=list(get('joint_ids'))
        if tuple(self.ids) != protocol.JOINT_IDS:
            raise ValueError('UART requires all seven logical joint IDs in order 1..7')
        if len(self.names)!=len(set(self.names)) or len(self.ids)!=len(set(self.ids)) or len(self.names)!=len(self.ids):
            raise ValueError('joint names and IDs must be unique and equally sized')
        arrays=[list(get(name)) for name in ('position_min','position_max','velocity_limit','torque_limit','kp_limit','kd_limit')]
        if any(len(values)!=len(self.ids) for values in arrays):
            raise ValueError('all limit arrays must match joint IDs')
        self.zero_tolerances = list(get('feedback_zero_tolerance_rad'))
        if (len(self.zero_tolerances) != 6 or
                any(not math.isfinite(v) or v < 0 for v in self.zero_tolerances)):
            raise ValueError('feedback_zero_tolerance_rad requires six nonnegative finite values')
        feedback_motor_ids = tuple(int(value) for value in get('feedback_motor_ids'))
        if len(feedback_motor_ids) != protocol.FIXED_AXIS_COUNT or any(not 1 <= value <= 255 for value in feedback_motor_ids):
            raise ValueError('feedback_motor_ids must contain seven valid CAN IDs')
        slot_joint_ids = protocol.validate_slot_joint_ids(get('slot_joint_ids'))
        slot_feedback_ids = tuple(int(value) for value in get('slot_feedback_ids'))
        if len(slot_feedback_ids) != protocol.FIXED_AXIS_COUNT or any(not 1 <= value <= 0x7ff for value in slot_feedback_ids):
            raise ValueError('slot_feedback_ids must contain seven valid CAN IDs')
        self.slot_feedback_ids = slot_feedback_ids
        limits={key:Limits(*(float(values[i]) for values in arrays)) for i,key in enumerate(self.ids)}
        source=str(get('command_source'))
        if source not in ('batch','per_joint'):
            raise ValueError('command_source must be batch or per_joint')
        ns=str(get('arm_namespace')).strip('/')
        prefix=f'/{ns}' if ns else ''
        # 8N1 每字节占 10 bit；反馈按同频率估算，两方向均留 20% 余量。
        for size in (protocol.COMMAND_SIZE, protocol.FEEDBACK_SIZE):
            if size * float(get('tx_hz')) * 10 > int(get('baudrate')) * .8:
                raise ValueError('UART frame rate exceeds 80% of line capacity')
        # 只有这一行实际打开 Linux 串口；link 线程随后负责所有读写。
        transport=PosixUart(port,int(get('baudrate')))
        try:
            link_options={key:get(key) for key in ('tx_hz','command_timeout','feedback_timeout','start_tolerance')}
            self.link=UartLink(transport,limits,expected_motor_ids=feedback_motor_ids,
                               slot_joint_ids=slot_joint_ids,**link_options)
        except BaseException:
            transport.close();raise
        group=ReentrantCallbackGroup()
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
        # batch 要求一条消息包含所有轴，per_joint 则由 link 按时间汇总。
        if source=='batch':
            self.create_subscription(Float64MultiArray,f'{prefix}/uart/mit_targets',self._batch,qos,callback_group=group)
        else:
            for name,key in zip(self.names,self.ids):
                self.create_subscription(JointMitCmd,f'{prefix}/joints/{name}/cmd/mit',
                    lambda msg,joint_id=key:self._single(joint_id,msg),qos,callback_group=group)
        self.g_open=float(get('gripper_open_rad'));self.g_close=float(get('gripper_close_rad'))
        self.g_travel=float(get('gripper_finger_travel_m'))
        if self.g_open==self.g_close:raise ValueError('invalid gripper position range')
        self.gripper_pub=self.create_publisher(JointMotorState,f'{prefix}/gripper/state',10) if 'gripper' in self.names else None
        self.joint_pub=self.create_publisher(JointState,f'{prefix}/joint_states',10)
        self.motor_pubs={key:self.create_publisher(JointMotorState,f'{prefix}/joints/{name}/state',10) for name,key in zip(self.names,self.ids)}
        status_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub=self.create_publisher(ArmStatus,f'{prefix}/arm_status',status_qos)
        self.diag_pub=self.create_publisher(String,f'{prefix}/uart/status',10)
        self.create_service(Trigger,f'{prefix}/enable',self._enable,callback_group=group)
        self.create_service(Trigger,f'{prefix}/disable',self._disable,callback_group=group)
        self.last_generation=0
        self.last_input_error=''
        self.last_reported_fault=''
        self.create_timer(.01,self._publish_feedback)
        self.create_timer(.1,self._publish_status)
        self.link.start()
        self.get_logger().info(f'UART {port}@{get("baudrate")}: source={source}, MIT {get("tx_hz")}Hz; explicit enable required')

    def _single(self,key,msg):
        """接收单轴 ROS MIT 指令并缓存，暂不直接写串口。"""
        try:
            self.link.set_targets([MitTarget(key,msg.pos,msg.vel,msg.kp,msg.kd,msg.tau)])
            self.last_input_error=''
        except ValueError as exc:
            self.last_input_error=str(exc)

    def _batch(self,msg):
        """接收按配置 ID 顺序排列的整组 [pos, vel, kp, kd, tau]。"""
        try:
            if len(msg.data)!=5*len(self.ids):
                raise ValueError('batch data must be [pos,vel,kp,kd,tau] for every configured joint')
            self.link.set_targets([MitTarget(key,*msg.data[5*i:5*i+5]) for i,key in enumerate(self.ids)],complete=True)
            self.last_input_error=''
        except ValueError as exc:
            self.last_input_error=str(exc)

    def _enable(self,request,response):
        """首帧写入即返回；七轴实际使能状态随后通过 arm_status 发布。"""
        response.success,response.message=self.link.request(True)
        return response

    def _disable(self,request,response):
        """ROS disable 服务；固定协议停止发包，由 STM32 看门狗完成失能。"""
        response.success,response.message=self.link.request(False)
        return response

    def _publish_feedback(self):
        """把最新 STATE 映射为 joint_states 和逐电机状态消息。"""
        snapshot=self.link.snapshot()
        feedback=snapshot['feedback']
        if feedback is None or snapshot['feedback_generation']==self.last_generation:
            return
        self.last_generation=snapshot['feedback_generation']
        # 历史测量不能冒充新数据，否则上游会错误地刷新反馈超时。
        if not snapshot['feedback_ready'] or any(not joint.feedback_fresh for joint in feedback.joints):
            return
        # 按 joint_id 映射，而不是假设 STM32 返回顺序与 ROS 顺序相同。
        values={j.joint_id:j for j in feedback.joints}
        msg=JointState()
        # 固定帧无 MCU 时间戳，这里仅记录主机发布时间。
        msg.header.stamp=self.get_clock().now().to_msg()
        # gripper 是电机名，不是 URDF 关节；其原始反馈走 gripper/state。
        # joint_states 只发布模型中的六轴和两个夹指，避免 MoveIt 持续报错。
        joints=[(name,key) for name,key in zip(self.names,self.ids) if name!='gripper']
        msg.name=[name for name,key in joints]
        msg.position=[_planning_position(values[key].pos, self.link.limits[key], tolerance)
                      for (name,key),tolerance in zip(joints,self.zero_tolerances)]
        msg.velocity=[values[key].vel for name,key in joints]
        msg.effort=[values[key].tau for name,key in joints]
        if self.gripper_pub is not None:
            idx=self.names.index('gripper');key=self.ids[idx];actual=values[key]
            slide=(actual.pos-self.g_close)/(self.g_open-self.g_close)*self.g_travel
            speed=actual.vel/(self.g_open-self.g_close)*self.g_travel
            msg.name=list(msg.name)+['gripper_joint1','gripper_joint2']
            msg.position=list(msg.position)+[-slide,slide]
            msg.velocity=list(msg.velocity)+[-speed,speed]
            msg.effort=list(msg.effort)+[math.nan,math.nan]
        self.joint_pub.publish(msg)
        for name,key in zip(self.names,self.ids):
            actual=values[key];motor=JointMotorState()
            motor.header=msg.header;motor.joint_name=name
            motor.position=actual.pos;motor.velocity=actual.vel
            motor.torque=actual.tau
            motor.temperature=actual.temperature;motor.mos_temperature=actual.mos_temperature
            # ROS 消息只有 8 位状态码；完整 uint16 保存在 uart/status JSON。
            motor.status_code=actual.status & 0xff
            self.motor_pubs[key].publish(motor)
            if name == "gripper":self.gripper_pub.publish(motor)

    def _publish_status(self):
        """发布主机状态、三组 flags 和原始电机状态。"""
        snapshot=self.link.snapshot();feedback=snapshot.pop('feedback')
        if snapshot['fault'] != self.last_reported_fault:
            if snapshot['fault']:
                self.get_logger().error(f"UART stopped: {snapshot['fault']}; {snapshot['fault_detail']}")
            self.last_reported_fault=snapshot['fault']
        state=ArmStatus()
        state.header.stamp=self.get_clock().now().to_msg()
        state.mode='mit';state.enabled=snapshot['enabled'];state.control_loop_active=snapshot['enabled']
        state.state_machine=snapshot['state'];state.joint_names=self.names
        if feedback is not None:
            values={j.joint_id:j for j in feedback.joints}
            state.per_joint_status_code=[values[key].status & 0xff for key in self.ids]
            state.per_joint_temperature=[values[key].temperature for key in self.ids]
            snapshot.update(feedback_state=feedback.state,system_fault=feedback.fault,
                            motor_status={name:values[key].status for name,key in zip(self.names,self.ids)},
                            group_flags=list(feedback.group_flags),
                            slot_joint_ids=list(self.link.slot_joint_ids),
                            slot_feedback_ids=list(self.slot_feedback_ids),
                            feedback_motor_ids=[joint.motor_id for joint in feedback.joints],
                            feedback_position_rad={name:values[key].pos for name,key in zip(self.names,self.ids)},
                            feedback_fresh={name:values[key].feedback_fresh for name,key in zip(self.names,self.ids)})
        errors=[value for value in (snapshot['fault'],self.last_input_error) if value]
        state.error_codes=errors
        snapshot['input_error']=self.last_input_error
        self.status_pub.publish(state)
        diag=String();diag.data=json.dumps(snapshot,ensure_ascii=False);self.diag_pub.publish(diag)

    def destroy_node(self):
        """先停止 UART worker，再释放 ROS 节点和串口。"""
        self.link.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node=None;executor=MultiThreadedExecutor(num_threads=3)
    try:
        node=UartBridge();executor.add_node(node);executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
