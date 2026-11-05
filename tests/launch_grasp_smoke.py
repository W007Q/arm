"""Exercise the installed unified simulation launch and task CLI, without hardware.

Run after sourcing ROS/install with a separate, nonzero ROS_DOMAIN_ID.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.action import get_action_server_names_and_types_by_node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('color', choices=['red', 'blue', 'yellow'], default='red', nargs='?')
    args = parser.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID', '0')) == 0:
        raise RuntimeError('use an isolated nonzero ROS_DOMAIN_ID')
    logfile = Path(f'/tmp/rebotarm_unified_launch_{os.environ["ROS_DOMAIN_ID"]}_{args.color}.log')
    with logfile.open('w') as log:
        process = subprocess.Popen(
            ['ros2', 'launch', 'rebotarm_bringup', 'grasp.launch.py', 'backend:=mujoco',
             'use_viewer:=false', 'detector_backend:=color'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        node = None
        try:
            rclpy.init()
            node = rclpy.create_node('unified_grasp_test_observer')
            expected = ['/rebotarm/move_to_pose', '/rebotarm/follow_joint_trajectory']
            deadline = time.monotonic() + 30.
            while True:
                assert process.poll() is None, logfile.read_text()
                providers = {name: [] for name in expected}
                for name, namespace in node.get_node_names_and_namespaces():
                    for action, _ in get_action_server_names_and_types_by_node(node, name, namespace):
                        if action in providers:
                            providers[action].append((namespace, name))
                if all(len(servers) == 1 for servers in providers.values()):
                    break
                if time.monotonic() > deadline:
                    raise AssertionError(providers)
                rclpy.spin_once(node, timeout_sec=.2)
            print('ACTION_PROVIDERS', providers, flush=True)
            result = subprocess.run(
                ['ros2', 'run', 'rebotarm_tasks', 'pick', '--color', args.color,
                 '--action-timeout-padding-sec', '40', '--ready-timeout-sec', '30'],
                capture_output=True, text=True, timeout=180.,
            )
            print(result.stdout, flush=True)
            if result.returncode:
                raise AssertionError(result.stderr + '\n' + logfile.read_text())
        finally:
            if node:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10.)


if __name__ == '__main__':
    main()
