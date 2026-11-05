"""RGB-D 时间同步回归；独立 ROS 域、合成图像，不连接相机或电机。"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

pytestmark = pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID', '0') == '0',
                                reason='use an isolated nonzero ROS_DOMAIN_ID')


def test_image_wait_does_not_block_late_tf_in_detector_main(tmp_path):
    """只发一帧图像；相应 TF 稍后到达时仍应完成检测。"""
    rclpy.init()
    observer = rclpy.create_node('rgbd_tf_test_observer')
    received = []
    observer.create_subscription(String, '/tf_test/vision/pingpong_ball',
                                 lambda msg: received.append(json.loads(msg.data)), 10)
    rgb_pub = observer.create_publisher(Image, '/tf_test/rgb', qos_profile_sensor_data)
    depth_pub = observer.create_publisher(Image, '/tf_test/depth', qos_profile_sensor_data)
    info_pub = observer.create_publisher(CameraInfo, '/tf_test/info', qos_profile_sensor_data)
    broadcaster = TransformBroadcaster(observer)
    params = dict(arm_namespace='tf_test', base_frame='tf_test_base', detector_backend='color',
                  image_topic='/tf_test/rgb', depth_topic='/tf_test/depth',
                  camera_info_topic='/tf_test/info', transform_timeout_sec='0.6')
    command = [sys.executable, '-c', 'from rebotarm_vision.rgbd_detector import main; main()', '--ros-args']
    for name, value in params.items():
        command.extend(['-p', f'{name}:={value}'])
    process = None
    log_path = tmp_path / 'detector.log'

    def spin_until(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert process.poll() is None, log_path.read_text()
            if predicate():
                return
            rclpy.spin_once(observer, timeout_sec=.01)
        assert predicate(), log_path.read_text()

    def spin_for(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=.01)

    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            spin_until(lambda: all(pub.get_subscription_count() for pub in (rgb_pub, depth_pub, info_pub)), 5.)
            stamp_ns = observer.get_clock().now().nanoseconds
            transform = TransformStamped()
            transform.header.frame_id = 'tf_test_base'
            transform.child_frame_id = 'tf_test_optical'
            transform.transform.rotation.w = 1.
            transform.header.stamp = Time(nanoseconds=stamp_ns - 100_000_000).to_msg()
            broadcaster.sendTransform(transform)
            spin_for(.1)

            rgb, depth, info = Image(), Image(), CameraInfo()
            for msg in (rgb, depth, info):
                msg.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
                msg.header.frame_id = 'tf_test_optical'
                msg.height = msg.width = 32
            rgb.encoding = 'rgb8'; rgb.step = 32 * 3
            rgb.data = np.zeros((32, 32, 3), dtype=np.uint8).tobytes()
            depth.encoding = '32FC1'; depth.step = 32 * 4
            depth.data = np.ones((32, 32), dtype='<f4').tobytes()
            info.k = [50., 0., 16., 0., 50., 16., 0., 0., 1.]
            rgb_pub.publish(rgb); depth_pub.publish(depth); info_pub.publish(info)
            # 图像回调此时正在等待这条较晚的 TF；单线程入口无法处理它。
            spin_for(.15)
            transform.header.stamp = Time(nanoseconds=stamp_ns + 100_000_000).to_msg()
            broadcaster.sendTransform(transform)
            spin_until(lambda: bool(received), 2.)
            assert received[0]['stamp_ns'] == stamp_ns
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=3.)
        observer.destroy_node()
        rclpy.shutdown()
