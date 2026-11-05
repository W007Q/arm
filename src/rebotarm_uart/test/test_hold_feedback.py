"""保持示例必须同时取得六轴和独立夹爪反馈，并检查各自新鲜度。"""
from types import SimpleNamespace

from rebotarm_uart import hold


def test_hold_waits_for_gripper_and_stops_if_only_arm_feedback_keeps_arriving(monkeypatch):
    clock = [10.]
    monkeypatch.setattr(hold.time, 'monotonic', lambda: clock[0])
    sent = []
    node = SimpleNamespace(names=[f'joint{i}' for i in range(1, 7)]+['gripper'],
                           measured={}, feedback_stamps={}, target=None,
                           kp=1., kd=.1, tau=[0.]*7,
                           get_logger=lambda: SimpleNamespace(info=lambda message: None),
                           pub=SimpleNamespace(publish=sent.append))
    arm = dict.fromkeys(node.names[:6], 0.)
    hold.HoldPublisher.capture(node, arm)
    assert node.target is None
    hold.HoldPublisher.capture(node, {'gripper': .1})
    hold.HoldPublisher.publish(node)
    assert len(sent) == 1 and len(sent[0].data) == 35
    clock[0] += .3
    hold.HoldPublisher.capture(node, arm)
    hold.HoldPublisher.publish(node)
    assert len(sent) == 1  # 夹爪反馈过期，不能用六轴的新消息掩盖。
