"""完整的轨迹→MIT→UART 回归，只连接 PTY 假 STM32。"""
import os
from pathlib import Path
import pty
import select
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from rebotarm_description.parameters import uart_parameters
from rebotarm_planning.task_server import RsTaskServer
from rebotarm_uart.bridge import UartBridge
from rebotarm_uart.posvel_to_mit import PosVelToMit
from rebotarm_uart import protocol as protocol
from peer import Peer

pytestmark = pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID', '0') == '0',
                                reason='use an isolated nonzero ROS_DOMAIN_ID')


def test_ros_enable_returns_after_first_write_without_waiting_for_feedback():
    """无 MCU 回包时服务仍立即确认发送；后台超时必须停止零输出流。"""
    master, slave = pty.openpty()
    rclpy.init(args=['--ros-args', '-p', f'port:={os.ttyname(slave)}',
                     '-p', 'arm_namespace:=enable_ack_test'])
    bridge = observer = executor = thread = None
    try:
        bridge = UartBridge()
        observer = Node('enable_ack_observer')
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(bridge)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        client = observer.create_client(Trigger, '/enable_ack_test/enable')
        assert client.wait_for_service(timeout_sec=3.)
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 1.
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=.01)
        assert future.done(), 'enable 服务仍在等待电机反馈'
        result = future.result()
        assert result.success and 'enable command sent' in result.message
        assert bridge.link.snapshot()['command_stream_active'] and not bridge.link.enabled
        assert bridge.link.feedback is None
        assert select.select([master], [], [], .3)[0]
        packets = protocol.Decoder(feedback=False).feed(os.read(master, 4096))
        assert packets
        for packet in packets:
            assert all(t.vel == t.kp == t.kd == t.tau == 0. for t in protocol.unpack_command(packet))
        deadline = time.monotonic() + 3.
        while not bridge.link.fault and time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=.01)
        assert 'post-enable feedback timeout' in bridge.link.fault
        count = bridge.link.tx_frames
        time.sleep(.06)
        assert bridge.link.tx_frames == count
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        if thread is not None:
            thread.join(2.)
        if bridge is not None:
            bridge.destroy_node()
        if observer is not None:
            observer.destroy_node()
        rclpy.shutdown()
        os.close(master)
        os.close(slave)


def test_slow_planning_keeps_uart_enabled_and_execution_finishes():
    master, slave = pty.openpty()
    scene = Path(__file__).resolve().parents[3] / 'src/rebotarm_description/mjcf/hardware_scene.xml'
    params = dict(port=os.ttyname(slave), backend='uart', arm_namespace='stream_test',
                  model_path=str(scene), kinematics_backend='mujoco',
                  hardware_scene_calibrated='true', d435_collision='false',
                  table_top_m='-0.0744', max_joint_speed='0.2')
    args = ['--ros-args']
    for key, value in params.items():
        args.extend(['-p', f'{key}:={value}'])
    rclpy.init(args=args)
    nodes, executors, threads = [], [], []
    peer = None
    observer = None
    try:
        bridge, converter, server = UartBridge(), PosVelToMit(), RsTaskServer()
        nodes = [bridge, converter, server]
        # 从现场零位读数启动；后续跟随实际编码后的绝对位置目标。
        peer = Peer(master, ids=tuple(uart_parameters()['feedback_motor_ids']),
                    follow_commands=True,
                    positions=[-.00019073777366429567]*5 +
                              [-.000572213320992887, -.00019073777366429567])
        # 与实际程序一致：桥和轨迹服务可等待，纯计算的适配器顺序运行。
        for node, executor in zip(nodes, [MultiThreadedExecutor(num_threads=3),
                                         SingleThreadedExecutor(),
                                         MultiThreadedExecutor(num_threads=4)]):
            executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            executors.append(executor); threads.append(thread); thread.start()
        observer = Node('execution_stream_test')

        def wait(predicate, timeout=6.):
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                rclpy.spin_once(observer, timeout_sec=.01)
            assert predicate(), bridge.link.snapshot()

        wait(lambda: bridge.link.snapshot()['targets_fresh'])
        assert not peer.commands
        ok, reason = bridge.link.request(True)
        assert ok, reason
        wait(lambda: server._arm_enabled)
        # 保持状态也需要连续发包，使用硬件配置中的命令超时。
        time.sleep(1.)
        assert bridge.link.enabled, bridge.link.snapshot()
        original_plan = server.planner.plan

        def slow_plan(*args, **kwargs):
            time.sleep(.6)  # 比上游/串口超时都长，模拟碰撞搜索耗时。
            return original_plan(*args, **kwargs)

        server.planner.plan = slow_plan
        client = ActionClient(observer, FollowJointTrajectory,
                              '/stream_test/follow_joint_trajectory')
        assert client.wait_for_server(timeout_sec=3.)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
        point = JointTrajectoryPoint()
        point.positions = [0., 0., .1, 0., 0., 0.]
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        sent = client.send_goal_async(goal)
        wait(sent.done)
        handle = sent.result()
        assert handle.accepted
        result = handle.get_result_async()
        wait(result.done, timeout=12.)
        assert result.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        assert bridge.link.enabled and not bridge.link.fault, bridge.link.snapshot()
        np.testing.assert_allclose(server._measured(), point.positions, atol=.002)
        # 真正的发布源消失时仍必须停发并锁定故障。
        executors[1].shutdown(timeout_sec=2.)
        threads[1].join(2.)
        wait(lambda: bridge.link.fault == 'upstream joint command timeout')
        assert not bridge.link.enabled
    finally:
        for executor in executors:
            executor.shutdown(timeout_sec=2.)
        for thread in threads:
            thread.join(2.)
        for node in nodes:
            node.destroy_node()
        if observer is not None:
            observer.destroy_node()
        if peer is not None:
            peer.close()
        rclpy.shutdown()
        os.close(master); os.close(slave)
