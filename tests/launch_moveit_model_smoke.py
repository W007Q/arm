"""离线验证新模型：MoveIt 规划 → 轨迹执行 → 关节状态确实改变。

先加载 ROS/install 环境，设置独立的非零 ROS_DOMAIN_ID。
mock 使用 ros2_control 假控制器；mujoco 使用共享 task_server 和物理仿真。
这两个选项都不会启动 UART/CAN 后端。
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('mock', 'mujoco'), default='mock')
    args = parser.parse_args()
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if domain == 0:
        raise RuntimeError('Set an isolated nonzero ROS_DOMAIN_ID first')
    logfile = Path(f'/tmp/rebotarm_model_{domain}_{args.backend}.log')
    processes = []
    latest = {}
    rclpy.init()
    node = rclpy.create_node('cad_model_smoke')
    topic = '/joint_states' if args.backend == 'mock' else '/cad_smoke/joint_states'
    node.create_subscription(JointState, topic,
                             lambda msg: latest.update(zip(msg.name, msg.position)),
                             qos_profile_sensor_data)

    def result(future, timeout=30.):
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        assert future.done(), f'timeout; see {logfile}'
        return future.result()

    with logfile.open('w') as log:
        def launch(package, filename, *arguments):
            processes.append(subprocess.Popen(
                ['ros2', 'launch', package, filename, *arguments],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))

        try:
            if args.backend == 'mock':
                launch('rebotarm_moveit_config', 'demo.launch.py', 'use_rviz:=false')
            else:
                launch('rebotarm_bringup', 'grasp.launch.py', 'backend:=mujoco',
                       'arm_namespace:=cad_smoke', 'use_viewer:=false', 'detector_backend:=color')
                launch('rebotarm_moveit_config', 'hardware.launch.py',
                       'arm_namespace:=cad_smoke', 'use_rviz:=false')
            validity = node.create_client(GetStateValidity, '/check_state_validity')
            planning = node.create_client(GetMotionPlan, '/plan_kinematic_path')
            execution = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
            assert validity.wait_for_service(timeout_sec=30.), logfile
            assert planning.wait_for_service(timeout_sec=10.), logfile
            assert execution.wait_for_server(timeout_sec=10.), logfile
            names = [f'joint{i}' for i in range(1, 7)]
            deadline = time.monotonic() + 10.
            while not all(name in latest for name in names) and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.1)
            assert all(name in latest for name in names), f'No joint states: {logfile}'
            request = GetStateValidity.Request()
            request.group_name = 'arm'
            request.robot_state.joint_state.name = names
            request.robot_state.joint_state.position = [latest[name] for name in names]
            request.robot_state.is_diff = True
            start = result(validity.call_async(request))
            assert start.valid, [(c.contact_body_1, c.contact_body_2) for c in start.contacts]
            print('START_VALID', start.valid, flush=True)

            request = GetMotionPlan.Request()
            motion = request.motion_plan_request
            motion.group_name = 'arm'
            motion.allowed_planning_time = 5.
            motion.num_planning_attempts = 3
            motion.max_velocity_scaling_factor = .2
            motion.max_acceleration_scaling_factor = .2
            motion.start_state.is_diff = True
            goal = [latest[name] for name in names]
            goal[2] = .12
            # A reachable Cartesian goal exercises the same IK path as dragging
            # RViz's interactive marker, rather than only planning joint targets.
            fk_client = node.create_client(GetPositionFK, '/compute_fk')
            ik_client = node.create_client(GetPositionIK, '/compute_ik')
            assert fk_client.wait_for_service(timeout_sec=5.)
            assert ik_client.wait_for_service(timeout_sec=5.)
            fk = GetPositionFK.Request()
            fk.header.frame_id = 'base_link'
            fk.fk_link_names = ['gripper_tcp']
            fk.robot_state.joint_state.name = names
            fk.robot_state.joint_state.position = goal
            target = result(fk_client.call_async(fk))
            assert target.error_code.val == 1
            pose = target.pose_stamped[0]
            ik = GetPositionIK.Request()
            ik.ik_request.group_name = 'arm'
            ik.ik_request.ik_link_name = 'gripper_tcp'
            ik.ik_request.pose_stamped = pose
            ik.ik_request.robot_state.joint_state.name = names
            ik.ik_request.robot_state.joint_state.position = [latest[name] for name in names]
            ik.ik_request.avoid_collisions = True
            ik.ik_request.timeout.sec = 2
            solution = result(ik_client.call_async(ik))
            assert solution.error_code.val == 1, (solution.error_code.val, logfile)
            print('CARTESIAN_IK_SUCCESS', flush=True)
            constraint = Constraints()
            position = PositionConstraint(header=pose.header, link_name='gripper_tcp', weight=1.)
            position.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[.001])]
            position.constraint_region.primitive_poses = [pose.pose]
            orientation = OrientationConstraint(header=pose.header, link_name='gripper_tcp',
                orientation=pose.pose.orientation, absolute_x_axis_tolerance=.01,
                absolute_y_axis_tolerance=.01, absolute_z_axis_tolerance=.01, weight=1.)
            constraint.position_constraints = [position]
            constraint.orientation_constraints = [orientation]
            motion.goal_constraints = [constraint]
            plan = result(planning.call_async(request)).motion_plan_response
            assert plan.error_code.val == 1, (plan.error_code.val, logfile)
            print('PLAN_SUCCESS', len(plan.trajectory.joint_trajectory.points), 'points', flush=True)
            command = ExecuteTrajectory.Goal()
            command.trajectory = plan.trajectory
            handle = result(execution.send_goal_async(command))
            assert handle.accepted
            completed = result(handle.get_result_async())
            assert completed.result.error_code.val == 1, (completed.result.error_code.val, logfile)
            deadline = time.monotonic() + 1.
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.1)
            assert latest['joint3'] > .05, latest
            fk.robot_state.joint_state.position = [latest[name] for name in names]
            measured = result(fk_client.call_async(fk))
            assert measured.error_code.val == 1
            actual = measured.pose_stamped[0].pose
            import math
            error_m = math.dist([actual.position.x, actual.position.y, actual.position.z],
                                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            a, b = actual.orientation, pose.pose.orientation
            error_rad = 2 * math.acos(min(1., abs(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)))
            assert error_m < .005 and error_rad < .03, (error_m, error_rad, latest)
            print('EXECUTE_SUCCESS joint3=', latest['joint3'],
                  'TCP_error_m=', error_m, 'rotation_error_rad=', error_rad, flush=True)
        finally:
            node.destroy_node()
            rclpy.shutdown()
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10.)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10.)


if __name__ == '__main__':
    main()
