"""Isolated-domain, hardware-free integration check; run with ROS_DOMAIN_ID set."""
import json
import sys
import threading
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rebotarm_mujoco_rs.mujoco_sync import RsMujocoSync
from rebotarm_mujoco_rs.scene_camera import RsSceneCamera
from rebotarm_vision.rgbd_detector import RgbdDetector
from rebotarm_planning.task_server import RsTaskServer
from rebotarm_tasks.gripper_state_node import GripperStateNode
from rebotarm_agent.rebotarm_mcp_server import ReBotArmToolBridge

color = sys.argv[1] if len(sys.argv)>1 else 'red'
rclpy.init(args=['--ros-args','-p','authority_mode:=true','-p','use_viewer:=false','-p','use_pose_topic:=true','-p','detection_source:=mujoco_rgbd','-p','detector_backend:=color','-p','gripper_kp:=120.0','-p','gripper_kd:=8.0','-p','gripper_force_limit:=2.0'])
nodes=[]
executor=MultiThreadedExecutor(num_threads=8)
camera_executor=SingleThreadedExecutor()
try:
    for cls in (RsMujocoSync, RsSceneCamera, RgbdDetector, RsTaskServer):
        node=cls();nodes.append(node)
        (camera_executor if cls is RsSceneCamera else executor).add_node(node)
    gripper_node=GripperStateNode();nodes.append(gripper_node);executor.add_node(gripper_node)
    bridge=ReBotArmToolBridge(namespace='rebotarm',motion_allowed=True,service_timeout_sec=5.,action_timeout_padding_sec=40.)
    nodes.append(bridge);executor.add_node(bridge)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    camera_thread=threading.Thread(target=camera_executor.spin,daemon=True);camera_thread.start()
    time.sleep(3.)
    print('DETECTIONS',json.dumps(bridge.detect_blocks(color)),flush=True)
    task=nodes[3]
    start=task._measured();goal=start.copy();goal[0]+=.08
    print('SMALL_MOTION',task._run_path([start,goal],[0.,1.]),flush=True)
    print('MEASURED_ERROR',max(abs(task._measured()-goal)),flush=True)
    result = bridge.pick_color(color,duration=2.)
    print('PICK',json.dumps(result),flush=True)
    if not result.get('ok'):
        import numpy as np
        with nodes[1]._lock:
            np.savez(f'/tmp/rebotarm_smoke_final_{color}.npz', qpos=nodes[1].data.qpos.copy())
        print('FINAL_SCENE',json.dumps(bridge._scene_payload),flush=True)
    assert result.get('ok'), result
finally:
    camera_executor.shutdown(timeout_sec=3.)
    executor.shutdown(timeout_sec=3.)
    for node in reversed(nodes):node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
