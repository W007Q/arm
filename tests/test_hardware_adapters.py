"""Command contracts with fake motor groups; never opens a hardware bus."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rebotarmcontroller.hardware_manager import HardwareManager
from rebotarm_uart.posvel_to_mit import PosVelToMit


def hardware():
    h = object.__new__(HardwareManager)
    h._cmd_lock = threading.RLock()
    h._robot = SimpleNamespace(has_gripper=True)
    h._gripper_target_position = 5.
    h._gripper_command_limit = .4
    h._gripper_torque_limit = .6
    h.gripper_open_position, h.gripper_close_position = 5., 0.
    h.get_gripper_state = lambda: (2., .1, .2, 0)
    sent = []
    h._gripper_group = SimpleNamespace(_mit_kp=[20.], _mit_kd=[2.], send_mit=lambda *args, **kwargs: sent.append((args, kwargs)))
    h._begin_gripper_command = lambda **kwargs: None
    h._begin_gripper_lowlevel = lambda mode: None
    return h, sent


def test_sdk_caps_total_pd_torque_and_zeroes_motor_gains():
    h, sent = hardware()
    h._send_gripper_bounded()
    _, command = sent[-1]
    assert command['tau'] == pytest.approx([.4])
    assert command['kp'] == pytest.approx([0.])
    assert command['kd'] == pytest.approx([0.])


def test_raw_sdk_mit_also_caps_pd_plus_feedforward():
    h, sent = hardware()
    h.send_gripper_mit_cmd(5., 0., 30., 2., 2.)
    assert sent[-1][1]['tau'] == pytest.approx([.6])
    assert sent[-1][1]['kp'] == pytest.approx([0.])
    with pytest.raises(ValueError):
        h.send_gripper_mit_cmd(6., 0., 30., 2., 2.)


def test_sdk_refuses_unlimited_pa43_posvel():
    h, sent = hardware()
    with pytest.raises(RuntimeError, match='torque limit'):
        h.send_gripper_pos_vel_cmd(3., .1)
    assert not sent


def converter():
    node = object.__new__(PosVelToMit)
    node.lock = threading.RLock()
    node.names = [f'joint{i}' for i in range(1, 7)] + ['gripper']
    node.measured = {name: (1., .1) for name in node.names}
    node.measured_at = time.monotonic()
    node.gripper_measured_at = node.measured_at
    node.feedback_timeout = .5
    node.target_timeout = .2
    node.targets, node.stamps = {}, {}
    node.speed_limits = dict(zip(node.names[:6], [.3]*5+[.5]))
    node.ever_commanded = False
    node.gripper_enabled = node.enabled = True
    node.g_open, node.g_close = 5., 0.
    node.g_kp, node.g_kd = 20., 2.
    node.g_limit = node.g_command_limit = .6
    node.g_target = None
    node.kp, node.kd = np.full(6, 10.), np.full(6, 1.)
    node.gravity_offset = np.zeros(6)
    node.gravity_direction = np.ones(6)
    node.gravity_scale = np.full(6, .3)
    node.gravity_limits = np.full(6, 1.)
    node.sdk = None
    node.published = []
    node.pub = SimpleNamespace(publish=node.published.append)
    return node


def test_uart_gravity_feedforward_applies_calibration_and_torque_caps():
    n = converter()
    n.gravity_offset = np.full(6, .5)
    n.gravity_direction = np.array([-1., 1., -1., 1., -1., 1.])
    n.gravity_scale = np.array([.1, .2, 0., .5, .1, .1])
    n.gravity_limits = np.array([.15, .5, .5, .2, .5, .5])
    model_inputs = []
    def gravity(q):
        model_inputs.append(q.copy())
        return np.array([2., 4., 6., -8., -10., 12.])
    n.sdk = SimpleNamespace(gravity=gravity)
    n.publish()
    np.testing.assert_allclose(model_inputs[0], [-.5, 1.5, -.5, 1.5, -.5, 1.5])
    assert list(n.published[-1].data[4:30:5]) == pytest.approx([-.15, .5, 0., -.2, .5, .5])
    # 标定映射只供模型使用；UART 位置仍保持原启动坐标。
    assert list(n.published[-1].data[:30:5]) == pytest.approx([1.]*6)


def test_uart_forwards_absolute_targets_and_limits_speed_for_firmware():
    n = converter()
    positions = [.1, -.2, .3, -.1, .2, -.3]
    n.on_arm_command(SimpleNamespace(positions=positions, velocity_limits=[.4]*6))
    assert len(set(n.stamps.values())) == 1  # 六轴在同一次回调里更新。
    n.publish()
    data = n.published[-1].data
    assert data[:30:5] == pytest.approx(positions)
    assert data[1:30:5] == pytest.approx([.3]*5+[.4])
    # vlim=0 明确暂停轨迹；无效限速不能覆盖上一条有效目标。
    n.on_arm_command(SimpleNamespace(positions=positions, velocity_limits=[0.]+[.4]*5))
    n.on_arm_command(SimpleNamespace(positions=[2.]*6, velocity_limits=[float('nan')]*6))
    n.publish()
    assert list(n.published[-1].data[:2]) == pytest.approx([.1, 0.])


def test_uart_acceptance_is_separate_from_reaching_and_output_is_bounded():
    n = converter()
    r = n.set_gripper(SimpleNamespace(position=4., max_effort=.3), SimpleNamespace())
    assert r.success and r.reached_position == 1.
    n.publish()
    assert len(n.published[-1].data) == 35
    assert n.published[-1].data[-5:] == pytest.approx([1., 0., 0., 0., .3])
    n.enabled = False
    assert not n.set_gripper(SimpleNamespace(position=4., max_effort=.3), SimpleNamespace()).success


def test_uart_stale_feedback_and_stale_arm_targets_stop_publication():
    n = converter()
    n.measured_at -= 1.
    n.publish()
    assert not n.published
    n.measured_at = time.monotonic()
    n.ever_commanded = True
    n.publish()
    assert not n.published


def test_uart_arm_feedback_does_not_refresh_separate_gripper_encoder():
    n = converter()
    n.measured_at = 0.
    n.gripper_measured_at = 0.
    n.on_state(SimpleNamespace(name=n.names[:6], position=[0.]*6, velocity=[0.]*6))
    assert n.measured_at > 0.
    assert n.gripper_measured_at == 0.
    n.publish()
    assert not n.published


def test_sdk_gripper_action_timeout_is_failure_not_success(monkeypatch):
    from rebotarmcontroller import ros_actions
    actions = object.__new__(ros_actions.ArmActions)
    actions._hardware = SimpleNamespace(
        set_gripper_target=lambda *args: None,
        get_gripper_state=lambda: (1., 0., 0., 0),
        gripper_reached_target=lambda: False,
    )
    elapsed = [0.]
    def tick():
        elapsed[0] += 1.
        return elapsed[0]
    monkeypatch.setattr(ros_actions, 'time', SimpleNamespace(monotonic=tick, sleep=lambda _: None))
    outcomes = []
    handle = SimpleNamespace(
        request=SimpleNamespace(command=SimpleNamespace(position=0., max_effort=0.)),
        is_cancel_requested=False, publish_feedback=lambda _: None,
        abort=lambda: outcomes.append('abort'), succeed=lambda: outcomes.append('succeed'),
    )
    result = actions.execute_gripper_command(handle)
    assert outcomes == ['abort']
    assert not result.reached_goal and not result.stalled
