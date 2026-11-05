"""把上游位置目标转换为 MIT 目标的可选 ROS 节点。

发送路径是 ``ArmPositionCommand -> on_arm_command -> publish ->
Float64MultiArray -> bridge``；它不打开 UART，也不直接接触 CAN。这里的
``positions`` 是六轴绝对目标，``velocity_limits`` 是非负限速。这与已烧录的 te 固件一致：
STM32 仍会按限速生成 5 ms 位置步进，速度填 0 会暂停位置轨迹。
夹爪的力矩由主机 PD 计算；可选重力前馈需要先标定模型零点和关节方向。
"""
import math
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from rebotarm_msgs.msg import ArmPositionCommand, JointMotorState, ArmStatus
from rebotarm_msgs.srv import SetGripper
from std_msgs.msg import Float64MultiArray
from rebotarm_description.command_interface import validate_position_command
from rebotarm_description.parameters import hardware_profile, load_hardware, uart_gains, uart_gravity, uart_parameters

class PosVelToMit(Node):
    """用实测关节状态保持安全初始目标，并发布完整 MIT 目标组。"""

    def __init__(self):
        super().__init__('rebotarm_posvel_to_mit')
        hardware = load_hardware()
        profile = hardware_profile(hardware)
        gains = uart_gains(hardware)
        gravity = uart_gravity(hardware)
        limits = uart_parameters(hardware)
        defaults=dict(arm_namespace='rebotarm',joint_names=[f'joint{i}' for i in range(1,7)]+['gripper'],
                      joint_ids=list(range(1,8)),kp=gains['kp'][0],kd=gains['kd'][0],arm_kp=gains['kp'][:6],
                      arm_kd=gains['kd'][:6],use_joint_gains=True,arm_velocity_limit=limits['velocity_limit'][:6],
                      gravity_joint_offset=gravity['joint_offset'],gravity_joint_direction=gravity['joint_direction'],
                      gravity_tau_scale=gravity['tau_scale'],gravity_torque_limit_nm=limits['torque_limit'][:6],
                      gravity_feedforward=False,sdk_urdf='',target_timeout=.35,
                      feedback_timeout=.2,gripper_enabled=True,gripper_open_rad=float(profile['gripper']['position_limits']['open']),
                      gripper_close_rad=float(profile['gripper']['position_limits']['close']),
                      gripper_kp=gains['kp'][6],gripper_kd=gains['kd'][6],
                      gripper_torque_limit_nm=float(profile['gripper']['torque_limit_nm']))
        for k,v in defaults.items(): self.declare_parameter(k,v)
        get=lambda k:self.get_parameter(k).value
        self.names=list(get('joint_names'));self.ids=list(get('joint_ids'))
        if len(self.names)!=len(self.ids) or len(set(self.ids))!=len(self.ids):
            raise ValueError('unique equally sized joint names/IDs required')
        self.gripper_enabled=bool(get('gripper_enabled'))
        if self.gripper_enabled and self.names != [f'joint{i}' for i in range(1,7)]+['gripper']:
            raise ValueError('seven-axis profile must name joint1..joint6, gripper')
        self.kp=np.array(get('arm_kp') if get('use_joint_gains') else [get('kp')]*6,dtype=float)
        self.kd=np.array(get('arm_kd') if get('use_joint_gains') else [get('kd')]*6,dtype=float)
        if self.kp.shape!=(6,) or self.kd.shape!=(6,) or not np.all(np.isfinite(np.r_[self.kp,self.kd])) or np.any(self.kp<0) or np.any(self.kd<0):
            raise ValueError('invalid MIT gains')
        speed_limits=np.array(get('arm_velocity_limit'),dtype=float)
        if speed_limits.shape!=(6,) or not np.all(np.isfinite(speed_limits)) or np.any(speed_limits<=0):
            raise ValueError('six positive arm velocity limits required')
        self.speed_limits=dict(zip(self.names[:6],speed_limits))
        self.gravity_offset=np.asarray(get('gravity_joint_offset'),dtype=float)
        self.gravity_direction=np.asarray(get('gravity_joint_direction'),dtype=float)
        self.gravity_scale=np.asarray(get('gravity_tau_scale'),dtype=float)
        self.gravity_limits=np.asarray(get('gravity_torque_limit_nm'),dtype=float)
        if (self.gravity_offset.shape!=(6,) or self.gravity_direction.shape!=(6,) or
                self.gravity_scale.shape!=(6,) or self.gravity_limits.shape!=(6,) or
                not np.all(np.isfinite(np.r_[self.gravity_offset,self.gravity_direction,self.gravity_scale,self.gravity_limits])) or
                not np.all(np.isin(self.gravity_direction, (-1., 1.))) or
                np.any(self.gravity_scale<0) or np.any(self.gravity_scale>1) or np.any(self.gravity_limits<=0)):
            raise ValueError('invalid UART gravity calibration parameters')
        self.target_timeout=float(get('target_timeout'));self.feedback_timeout=float(get('feedback_timeout'))
        self.g_open=float(get('gripper_open_rad'));self.g_close=float(get('gripper_close_rad'))
        self.g_kp=float(get('gripper_kp'));self.g_kd=float(get('gripper_kd'))
        self.g_limit=float(get('gripper_torque_limit_nm'));self.g_command_limit=self.g_limit
        self.g_target=None;self.enabled=False;self.ever_commanded=False
        self.lock=threading.RLock();self.targets={};self.stamps={};self.measured={};self.measured_at=0.
        self.gripper_measured_at=0.
        self.sdk=None
        if get('gravity_feedforward'):
            from rebotarm_planning.sdk_model import SdkRobotModel
            self.sdk=SdkRobotModel(urdf_path=str(get('sdk_urdf')) or None)
        ns=str(get('arm_namespace')).strip('/');prefix=f'/{ns}'
        group=ReentrantCallbackGroup()
        self.command_topic = f'{prefix}/arm/command/position'
        self.create_subscription(ArmPositionCommand, self.command_topic,
                                 self.on_arm_command, 10, callback_group=group)
        self.create_subscription(JointState,f'{prefix}/joint_states',self.on_state,qos_profile_sensor_data,callback_group=group)
        self.create_subscription(ArmStatus,f'{prefix}/arm_status',self.on_status,10,callback_group=group)
        if self.gripper_enabled:
            self.create_subscription(JointMotorState,f'{prefix}/gripper/state',self.on_gripper_state,10,callback_group=group)
            self.create_service(SetGripper,f'{prefix}/gripper/set',self.set_gripper,callback_group=group)
        self.pub=self.create_publisher(Float64MultiArray,f'{prefix}/uart/mit_targets',1)
        # UART 只发送 tx_hz 帧；更高的 ROS 频率只会增加排队和调度负担。
        self.create_timer(1./float(limits['tx_hz']),self.publish,callback_group=group)

    def on_arm_command(self, msg):
        """缓存六轴批量目标；同一时间戳保证 UART 不会拼出半组目标。"""
        try:
            positions, speeds = validate_position_command(msg.positions, msg.velocity_limits)
        except ValueError:
            return
        now=time.monotonic()
        with self.lock:
            for name,pos,speed in zip(self.names[:6],positions,speeds):
                self.targets[name]=(float(pos),min(float(speed),self.speed_limits[name]))
                self.stamps[name]=now
            self.ever_commanded=True
    def on_state(self,msg):
        """缓存 ROS 反馈；发布周期只使用新鲜且包含全部配置轴的测量值。"""
        if not all(n in msg.name for n in self.names[:6]):return
        state={n:(p,v) for n,p,v in zip(msg.name,msg.position,msg.velocity)}
        if not all(n in state and all(math.isfinite(x) for x in state[n]) for n in self.names[:6]):return
        with self.lock:
            self.measured.update({n:state[n] for n in self.names[:6]})
            self.measured_at=time.monotonic()
    def on_gripper_state(self,msg):
        """夹爪电机角度单独接收，不混入只包含 URDF 关节的 joint_states。"""
        if not all(math.isfinite(v) for v in (msg.position,msg.velocity)):return
        with self.lock:
            self.measured['gripper']=(msg.position,msg.velocity)
            self.gripper_measured_at=time.monotonic()
    def on_status(self,msg):
        """失能后清掉任务目标，避免重新使能时沿用旧指令。"""
        with self.lock:
            self.enabled=bool(msg.enabled)
            if not self.enabled:
                self.g_target=None
                self.targets.clear(); self.stamps.clear(); self.ever_commanded=False
    def set_gripper(self,req,res):
        """校验夹爪目标并记录，实际力矩在 ``publish`` 中按实测位置计算。"""
        with self.lock:
            lo,hi=sorted((self.g_close,self.g_open))
            valid=(self.enabled and self.g_limit>0 and math.isfinite(req.position)
                   and lo<=req.position<=hi and math.isfinite(req.max_effort) and req.max_effort>=0
                   and time.monotonic()-self.measured_at<self.feedback_timeout
                   and time.monotonic()-self.gripper_measured_at<self.feedback_timeout)
            res.success=bool(valid)
            if valid:
                self.g_target=float(req.position)
                self.g_command_limit=min(float(req.max_effort),self.g_limit) if req.max_effort>0 else self.g_limit
            res.reached_position=float(self.measured.get('gripper',(0.,0.))[0])
        return res
    def publish(self):
        """按 UART 帧率发布整组 MIT 数据；反馈或目标过期时停止发布。"""
        now=time.monotonic()
        with self.lock:
            if now-self.measured_at>self.feedback_timeout or not self.measured:return
            if self.gripper_enabled and (now-self.gripper_measured_at>self.feedback_timeout or 'gripper' not in self.measured):return
            if self.ever_commanded and any(n not in self.stamps or now-self.stamps[n]>self.target_timeout for n in self.names[:6]):return
            # 首个任务目标到达前，用实测位置填充目标，便于安全完成 enable 对齐。
            targets=[self.targets.get(n,(self.measured[n][0],0.)) for n in self.names[:6]]
            measured_q=np.array([self.measured[n][0] for n in self.names[:6]])
            # tau 是可选的主机重力前馈；STM32 不应再次叠加同一补偿。
            if self.sdk:
                # q_model 对齐 URDF；方向变换后再把模型力矩变回电机方向。
                q_model=measured_q*self.gravity_direction+self.gravity_offset
                tau=self.sdk.gravity(q_model)*self.gravity_direction*self.gravity_scale
                if tau.shape!=(6,) or not np.all(np.isfinite(tau)):
                    raise ValueError('gravity model must return six finite torques')
                tau=np.clip(tau,-self.gravity_limits,self.gravity_limits)
            else:
                tau=np.zeros(6)
            data=[]
            for i,(pos,speed) in enumerate(targets):
                data.extend([pos,float(speed),float(self.kp[i]),float(self.kd[i]),float(tau[i])])
            if self.gripper_enabled:
                pos,vel=self.measured['gripper']
                target=pos if self.g_target is None or not self.enabled else self.g_target
                torque=float(np.clip(self.g_kp*(target-pos)-self.g_kd*vel,-self.g_command_limit,self.g_command_limit))
                data.extend([pos,0.,0.,0.,torque])
        msg=Float64MultiArray();msg.data=data;self.pub.publish(msg)

def main(args=None):
    # 回调均不等待硬件 I/O；顺序处理可避免多个高频回调争抢同一把锁。
    rclpy.init(args=args);node=PosVelToMit();ex=SingleThreadedExecutor();ex.add_node(node)
    try:ex.spin()
    finally:ex.shutdown();node.destroy_node();rclpy.shutdown()
