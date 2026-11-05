"""Requires sourced ROS Humble plus rebotarm_msgs; talks only to a PTY peer."""
import json
import os
import pty
import threading
import time

import pytest
rclpy=pytest.importorskip('rclpy')
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64MultiArray,String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from rebotarm_msgs.msg import JointMitCmd, JointMotorState
from rebotarm_uart.bridge import UartBridge, _planning_position
from peer import Peer
from test_uart_link import wait_for

pytestmark = pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID', '0') == '0',
                                reason='use an isolated nonzero ROS_DOMAIN_ID')


def test_planning_position_aligns_only_one_uart_step_at_joint_boundary():
    from rebotarm_uart.link import Limits
    j3 = Limits(0., 2.0943951, .3, 1.)
    assert _planning_position(-.0001907378, j3) == 0.
    assert _planning_position(-.0006, j3) == pytest.approx(-.0006)
    assert _planning_position(.02, j3) == .02


def test_confirmed_j5_zero_error_is_bounded_and_opt_in():
    from rebotarm_uart.link import Limits
    j5 = Limits(-1.5707963, 1.5707963, .3, 3.)
    for value in (-.01, -.0074387733, .0074387733, .01):
        assert _planning_position(value, j5, .01) == 0.
    assert _planning_position(.0101, j5, .01) == .0101
    assert _planning_position(-.0101, j5, .01) == -.0101
    assert _planning_position(.0074387733, j5) == .0074387733


@pytest.mark.parametrize('source',['batch','per_joint'])
def test_ros_to_uart_and_back(source):
    axes=7
    master,slave=pty.openpty()
    extra=[]
    names=[f'joint{i}' for i in range(1,7)]+['gripper']
    # Use targets within the actual arm limits (joint2's upper bound is zero).
    positions=[.01,-.02,.03,.04,.05,.06,.07]
    if axes==7:
        for key, value in dict(joint_names=names, joint_ids=list(range(1,8)),
                              position_min=[-3.14]*7,position_max=[5.0]*7,
                              velocity_limit=[2.0]*7,torque_limit=[27.0]*7,
                              kp_limit=[500.0]*7,kd_limit=[20.0]*7).items():
            extra.extend(['-p',f'{key}:={json.dumps(value)}'])
    rclpy.init(args=['--ros-args' ,'-p',f'port:={os.ttyname(slave)}','-p','tx_hz:=50.0',
                    '-p','arm_namespace:=uart_bridge_test',
                    '-p',f'command_source:={source}','-p','command_timeout:=0.15']+extra)
    bridge=None;client=None;peer=None;thread=None
    executor=MultiThreadedExecutor(num_threads=4)
    try:
        bridge=UartBridge()
        # J5 在第六个 UART 槽；验证模型视为零，但原始诊断和电机反馈不被改写。
        peer=Peer(master, ids=bridge.link.expected_motor_ids, positions=[0.] * 5 + [.0074387733, 0.])
        client=Node('uart_test_observer')
        measured=[];diagnostics=[];gripper_measured=[]
        client.create_subscription(JointState,'/uart_bridge_test/joint_states',measured.append,10)
        client.create_subscription(JointMotorState,'/uart_bridge_test/gripper/state',gripper_measured.append,10)
        client.create_subscription(String,'/uart_bridge_test/uart/status',lambda msg:diagnostics.append(json.loads(msg.data)),10)
        enable=client.create_client(Trigger,'/uart_bridge_test/enable')
        disable=client.create_client(Trigger,'/uart_bridge_test/disable')
        if source=='batch':
            pub=client.create_publisher(Float64MultiArray,'/uart_bridge_test/uart/mit_targets',1)
            def publish():
                msg=Float64MultiArray();msg.data=[v for i in range(1,axes+1) for v in (positions[i-1],.02*i,12.,1.,.1*i)]
                pub.publish(msg)
        else:
            pubs=[client.create_publisher(JointMitCmd,f'/uart_bridge_test/joints/{name}/cmd/mit',1) for name in names]
            def publish():
                for i,pub in enumerate(pubs,1):
                    msg=JointMitCmd();msg.pos=positions[i-1];msg.vel=.02*i;msg.kp=12.;msg.kd=1.;msg.tau=.1*i
                    pub.publish(msg)
        timer=client.create_timer(.02,publish)
        executor.add_node(bridge);executor.add_node(client)
        thread=threading.Thread(target=executor.spin);thread.start()
        wait_for(lambda:measured and bridge.link.snapshot()['targets_fresh'],timeout=5.)
        assert not peer.commands
        assert enable.wait_for_service(timeout_sec=2.)
        future=enable.call_async(Trigger.Request());wait_for(future.done,timeout=3.)
        assert future.result().success,future.result().message
        wait_for(lambda:len(peer.commands)>=8,timeout=3.)
        assert list(measured[-1].position)==[0.]*(axes+1 if axes==7 else axes)
        # Peer values follow wire slots; published state follows logical joints.
        assert list(measured[-1].velocity)[:6]==pytest.approx([.03,.04,.05,.01,.06,.02])
        if axes==7:
            assert measured[-1].name[-2:]==['gripper_joint1','gripper_joint2']
            assert 'gripper' not in measured[-1].name
            assert gripper_measured[-1].velocity==pytest.approx(.07)
        from rebotarm_uart.protocol import unpack_command
        targets=unpack_command(peer.commands[-1])
        assert targets[0].tau==pytest.approx(.1)
        assert targets[5].pos==pytest.approx(.06)
        assert len(targets)==axes
        if axes==7:assert targets[-1].pos==pytest.approx(.07)
        assert list(measured[-1].effort)[:6]==pytest.approx([.06,.08,.1,.02,.12,.04])
        assert diagnostics[-1]['group_flags']==[3,3,3]
        raw_positions=dict.fromkeys(names,0.);raw_positions['joint5']=.0074387733
        assert diagnostics[-1]['feedback_position_rad']==pytest.approx(raw_positions)
        assert 'applied_command_seq' not in diagnostics[-1]
        timer.cancel()
        wait_for(lambda:any('upstream' in d['fault'] for d in diagnostics),timeout=3.)
        future=disable.call_async(Trigger.Request());wait_for(future.done,timeout=3.)
        assert future.result().success
    finally:
        executor.shutdown(timeout_sec=2.)
        if thread:thread.join(2.)
        if bridge:bridge.destroy_node()
        if client:client.destroy_node()
        if peer:peer.close()
        rclpy.shutdown();os.close(master);os.close(slave)
