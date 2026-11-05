import os
import pty
import struct
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rebotarm_uart import protocol as p
from rebotarm_uart.link import UartLink, Limits
from rebotarm_uart.transport import PosixUart
from rebotarm_description.parameters import uart_parameters
from peer import Peer, feedback_bytes


def wait_for(predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


def enable_and_confirm(link, timeout=1.):
    """服务先确认首帧发出；运行类测试另外等待真实对端反馈。"""
    ok, reason = link.request(True, timeout=timeout)
    assert ok, reason
    wait_for(lambda: link.enabled or bool(link.fault), timeout=timeout + .2)
    assert link.enabled, link.snapshot()


@pytest.fixture
def connection():
    master, slave = pty.openpty()
    link = UartLink(PosixUart(os.ttyname(slave), 115200),
                    {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)}, feedback_timeout=.08,
                    expected_motor_ids=(1, 2, 8, 9, 16, 1, 2))
    peer = Peer(master)
    targets = [p.MitTarget(i, .01 * i, .02, 1.2, .6, .1) for i in range(1, 8)]
    stop = threading.Event()
    publishing = threading.Event()
    publishing.set()

    def publish():
        while not stop.wait(.02):
            if publishing.is_set():
                link.set_targets(targets, complete=True)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    link.start()
    wait_for(lambda: link.snapshot()['feedback_ready'] and link.snapshot()['targets_fresh'])
    yield link, peer, targets, publishing
    stop.set()
    thread.join(1.)
    link.close()
    peer.close()
    os.close(master)
    os.close(slave)
    assert peer.error is None


def test_explicit_enable_streams_absolute_targets_then_disable_stops(connection):
    link, peer, targets, _ = connection
    assert not peer.commands
    enable_and_confirm(link)
    wait_for(lambda: len(peer.commands) >= 8)
    assert p.unpack_command(peer.commands[-1]) == tuple(targets)
    assert len(peer.commands[-1]) == 72
    assert link.request(False)[0]
    count = len(peer.commands)
    time.sleep(.55)
    assert len(peer.commands) == count
    wait_for(lambda: link.snapshot()['feedback'].state == 0)


def test_identical_targets_keep_running_but_stale_inputs_latch_fault(connection):
    link, peer, _, publishing = connection
    enable_and_confirm(link)
    wait_for(lambda: len(peer.commands) >= 12)
    assert link.snapshot()['enabled']
    publishing.clear()
    wait_for(lambda: 'upstream joint command timeout' in link.snapshot()['fault'])
    detail = link.snapshot()['fault_detail']
    assert max(detail['command_age_sec'].values()) > link.command_timeout
    assert detail['feedback_age_sec'] < link.feedback_timeout
    count = len(peer.commands)
    time.sleep(.06)
    assert len(peer.commands) == count
    assert not link.request(True)[0]
    assert link.request(False)[0]
    assert not link.snapshot()['fault_detail']


def test_feedback_loss_stops_stream(connection):
    link, peer, _, _ = connection
    enable_and_confirm(link)
    peer.feedback_enabled = False
    wait_for(lambda: 'UART feedback timeout' in link.snapshot()['fault'])
    assert not link.snapshot()['enabled']


@pytest.mark.parametrize('failure', ['stale', 'group1', 'group2', 'group3', 'fault', 'motorevo'])
def test_motor_or_group_failure_stops_stream(connection, failure):
    link, peer, _, _ = connection
    enable_and_confirm(link)
    if failure == 'stale':
        peer.stale_joint = 7
    elif failure == 'fault':
        peer.fault_joint = 3
    elif failure == 'motorevo':
        peer.motorevo_status = 0x8001
    else:
        flags = [3, 3, 3]
        flags[int(failure[-1]) - 1] = 0
        peer.flags = flags
    wait_for(lambda: bool(link.snapshot()['fault']))
    assert not link.snapshot()['enabled']


def test_uninitialized_group_does_not_block_send_but_never_confirms_running(connection):
    link, peer, _, _ = connection
    peer.flags = (0, 0x80, 0)
    wait_for(lambda: link.snapshot()['feedback'].group_flags[1] == 0x80)
    # 初始化位不阻止服务发首帧；运行确认在服务返回后独立超时。
    ok, reason = link.request(True, timeout=.15)
    assert ok and 'enable command sent' in reason
    wait_for(lambda: bool(peer.commands))
    wait_for(lambda: 'post-enable feedback timeout' in link.fault)
    assert not link.enabled


@pytest.mark.parametrize('stale', [True, False])
@pytest.mark.parametrize('motor_id', [0x10, 0x0a])
def test_cached_partial_enable_bits_do_not_block_the_first_write(stale, motor_id):
    # 启动前只有 J3 使能的缓存，不论新鲜与否都不能阻止首帧。
    motor_ids = (1, 2, 8, 9, motor_id, 1, 2)
    raw = bytearray(feedback_bytes(ids=motor_ids))
    j3_slot = 4  # zero-based wire slot 5 -> logical joint3
    raw[1 + 18 * j3_slot + 1] = 1
    raw[1 + 18 * j3_slot + 2] = int(not stale)
    packets = []
    link = UartLink(SimpleNamespace(write=packets.append), {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=motor_ids)
    link._receive(raw)
    ok, reason = link.request(True, timeout=.5)
    assert ok and 'enable command sent' in reason
    assert not link.enabled and link.pending is not None
    assert len(packets) == 1
    assert all(t == p.MitTarget(t.joint_id, 0., 0., 0., 0., 0.) for t in p.unpack_command(packets[0]))
    link._receive(feedback_bytes(running=True, ids=motor_ids))
    assert link.enabled and link.pending is None


def test_stale_idle_feedback_recovers_after_zero_output_handshake(connection):
    link, peer, targets, _ = connection
    peer.stale_when_idle = True
    peer.enable_delay = .08
    wait_for(lambda: not any(j.feedback_fresh for j in link.snapshot()['feedback'].joints))
    enable_and_confirm(link)
    wait_for(lambda: p.unpack_command(peer.commands[-1]) == tuple(targets))
    # Multiple stale UART reports during ENABLING must not abort the handshake.
    assert len(peer.commands) >= 4
    for packet in peer.commands[:3]:
        assert all(t == p.MitTarget(t.joint_id, 0., 0., 0., 0., 0.) for t in p.unpack_command(packet))


def test_enable_can_bootstrap_without_upstream_measurements(connection):
    link, peer, _, publishing = connection
    publishing.clear()
    peer.stale_when_idle = True
    with link.lock:
        link.commands.clear()
    wait_for(lambda: not any(j.feedback_fresh for j in link.snapshot()['feedback'].joints))
    assert not link.snapshot()['targets_fresh']
    enable_and_confirm(link)
    # The temporary zero-output hold must expire unless the real publisher takes over.
    wait_for(lambda: 'upstream joint command timeout' in link.snapshot()['fault'])
    assert all(t.kp == t.kd == t.tau == t.vel == 0.
               for packet in peer.commands for t in p.unpack_command(packet))


def test_missing_motor_feedback_during_enable_times_out_and_stops(connection):
    link, peer, _, _ = connection
    peer.stale_joint = 7
    wait_for(lambda: not all(j.feedback_fresh for j in link.snapshot()['feedback'].joints))
    ok, reason = link.request(True, timeout=.15)
    assert ok and 'enable command sent' in reason
    wait_for(lambda: 'post-enable feedback timeout' in link.fault)
    assert 'stale logical joints=[7]' in link.fault
    assert link.snapshot()['fault_detail']['stale_logical_joints'] == [7]
    assert peer.commands and not link.snapshot()['enabled']
    count = link.tx_frames
    time.sleep(.06)
    assert link.tx_frames == count
    assert all(t.kp == t.kd == t.tau == t.vel == 0.
               for packet in peer.commands for t in p.unpack_command(packet))


def test_new_measurement_checks_target_alignment_before_normal_output(connection):
    link, peer, targets, publishing = connection
    publishing.clear()
    with link.lock:
        link.set_targets([replace(t, pos=1.) for t in targets], complete=True)
    ok, reason = link.request(True, timeout=.5)
    assert ok and 'enable command sent' in reason
    wait_for(lambda: 'initial targets too far' in link.fault)
    assert peer.commands
    assert all(t.kp == t.kd == t.tau == 0.
               for packet in peer.commands for t in p.unpack_command(packet))


def test_enable_timeout_does_not_leave_stream_running(connection):
    link, peer, _, _ = connection
    peer.accept_enable = False
    ok, reason = link.request(True, timeout=.15)
    assert ok and 'enable command sent' in reason
    wait_for(lambda: 'post-enable feedback timeout' in link.fault)
    assert not link.snapshot()['enabled']


def test_enable_writes_first_frame_and_returns_without_any_feedback_or_worker():
    packets = []
    now = [10.]
    link = UartLink(SimpleNamespace(write=packets.append),
                    {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=(1, 2, 8, 9, 0x14, 1, 2), clock=lambda: now[0])
    # 没有 worker、没有 read 接口、没有任何反馈，首帧仍必须直接发出并返回。
    ok, reason = link.request(True, timeout=.2)
    assert ok and 'enable command sent' in reason
    assert len(packets) == link.tx_frames == 1
    assert all(t == p.MitTarget(t.joint_id, 0., 0., 0., 0., 0.) for t in p.unpack_command(packets[0]))
    snapshot = link.snapshot()
    assert snapshot['command_stream_active'] and not snapshot['enabled']
    assert not snapshot['feedback_ready'] and snapshot['state'] == 'ENABLING'
    assert link.request(True)[0] and len(packets) == 1  # 重复请求不重启确认期限。
    now[0] += .02
    link._tick(now[0])
    assert len(packets) == 2 and packets[0] == packets[1]
    now[0] += .2
    link._tick(now[0])
    assert 'post-enable feedback timeout; no recent 131-byte UART report' == link.fault
    assert not link.snapshot()['command_stream_active']
    link._tick(now[0] + 1.)
    assert len(packets) == 2


@pytest.mark.parametrize('error', [OSError('port disconnected'), TimeoutError('write timeout')])
def test_enable_write_failure_is_reported_without_claiming_command_was_sent(error):
    def failed_write(_):
        raise error

    link = UartLink(SimpleNamespace(write=failed_write),
                    {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=(1, 2, 8, 9, 0x14, 1, 2))
    ok, reason = link.request(True)
    assert not ok and 'UART transport error' in reason
    assert link.tx_frames == 0 and link.pending is None
    assert not link.snapshot()['command_stream_active'] and not link.enabled


def test_cached_running_feedback_cannot_confirm_a_new_enable_request():
    packets = []
    ids = (1, 2, 8, 9, 0x14, 1, 2)
    link = UartLink(SimpleNamespace(write=packets.append),
                    {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)}, expected_motor_ids=ids)
    link._receive(feedback_bytes(running=True, ids=ids))
    assert link.request(True)[0]
    assert len(packets) == 1 and not link.enabled
    link._receive(feedback_bytes(running=True, ids=ids))
    assert link.enabled and link.pending is None


def test_targets_reject_invalid_values_before_caching(connection):
    link, _, targets, publishing = connection
    publishing.clear()
    with link.lock:
        before = dict(link.commands)
        for bad in ([replace(targets[0], kp=150.)], [replace(targets[0], pos=4.)],
                    targets + [targets[0]], [replace(targets[0], joint_id=8)]):
            with pytest.raises(ValueError):
                link.set_targets(bad)
            assert link.commands == before
        with pytest.raises(ValueError):
            link.set_targets(targets[:-1], complete=True)


@pytest.mark.parametrize('lower,upper,pos,expected', [
    (0., 2.0943951, -12.5 / 65535, 0.),  # J3: CAN zero is slightly negative.
    (-2.6179939, 0., 12.5 / 65535, 0.),  # J2: CAN zero is slightly positive.
    (0., 5., -12.5 / 65535, 0.),         # Gripper: same one-sided lower bound.
    (-2.6179939, 0., -2.6179939, -2.617),
    (-3.1415927, 3.1415927, -3.1415927, -3.141),
    (-3.1415927, 3.1415927, 3.1415927, 3.141),
])
def test_position_rounding_keeps_actual_wire_target_inside_limits(lower, upper, pos, expected):
    target = p.MitTarget(3, pos, 0., 0., 0., 0.)
    checked = Limits(lower, upper, .3, 1.).check(target)
    targets = [replace(checked, joint_id=i) for i in range(1, 8)]
    sent = p.unpack_command(p.pack_command(targets))[2]
    assert sent.pos == expected
    assert lower <= sent.pos <= upper
    assert target.pos == pos  # Do not overwrite the measured position.


@pytest.mark.parametrize('field,value,label', [
    ('pos', -.00051, 'pos'), ('pos', 2.095, 'pos'),
    ('vel', .301, 'vlim'), ('tau', 1.001, 'tau'), ('kp', 3.2, 'kp'), ('kd', 1.5, 'kd'),
])
def test_real_limit_violations_report_the_field_value_and_range(field, value, label):
    target = p.MitTarget(3, .2, .2, 3.1, 1.4, 1.)
    with pytest.raises(ValueError) as error:
        Limits(0., 2.0943951, .3, 1., 3.1, 1.4).check(replace(target, **{field: value}))
    assert f'joint 3 exceeds configured limits: {label}={value}' in str(error.value)
    assert 'allowed=[' in str(error.value)


@pytest.mark.parametrize('j3_pos,success', [(-12.5 / 65535, True), (-.01, False)])
def test_enable_with_real_limits_distinguishes_can_zero_from_real_position_error(j3_pos, success):
    params = uart_parameters()
    limits = {key: Limits(*(params[name][key - 1] for name in
                           ('position_min', 'position_max', 'velocity_limit', 'torque_limit', 'kp_limit', 'kd_limit')))
              for key in range(1, 8)}
    packets = []
    link = UartLink(SimpleNamespace(write=packets.append), limits,
                    expected_motor_ids=(1, 2, 8, 9, 16, 1, 2))
    link._receive(feedback_bytes(fresh=False))
    ok, reason = link.request(True, timeout=1.)
    assert ok and 'enable command sent' in reason
    try:
        assert link.pending is not None and len(packets) == 1
        raw = bytearray(feedback_bytes(running=True))
        positions = {2: 12.5 / 65535, 3: j3_pos, 7: -12.5 / 65535}
        for joint, pos in positions.items():
            struct.pack_into('<f', raw, 1 + 18 * p.SLOT_JOINT_IDS.index(joint) + 6, pos)
        link._receive(raw)
        assert link.enabled == success
        assert next(j for j in link.feedback.joints if j.joint_id == 3).pos == pytest.approx(j3_pos)
        if success:
            # The normal adapter also uses measurements as initial targets.
            link.set_targets([p.MitTarget(key, positions.get(key, 0.), 0., 1.2, .3, 0.)
                              for key in range(1, 8)], complete=True)
            link._tick(time.monotonic())
            assert all(t.pos == 0. for t in p.unpack_command(packets[-1]))
            assert not link.fault
        else:
            assert 'joint 3 exceeds configured limits: pos=' in link.fault
            assert 'allowed=[0, 2.0943951]' in link.fault
            link._tick(time.monotonic())
            assert len(packets) == 1  # Only the zero-output handshake was sent.
    finally:
        link.request(False)


def test_disable_can_cancel_pending_enable(connection):
    link, peer, _, _ = connection
    peer.accept_enable = False
    assert link.request(True)[0]
    assert link.pending is not None
    assert link.request(False)[0]
    assert link.pending is None
    count = link.tx_frames
    time.sleep(.06)
    assert link.tx_frames == count
    assert not link.snapshot()['enabled']


def test_serial_disconnect_latches_transport_fault():
    master, slave = pty.openpty()
    link = UartLink(PosixUart(os.ttyname(slave)), {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=(1, 2, 8, 9, 16, 1, 2))
    link.start()
    os.close(master)
    try:
        wait_for(lambda: 'transport error' in link.snapshot()['fault'])
        assert not link.snapshot()['enabled']
    finally:
        link.close()
        os.close(slave)


def test_enable_timeout_records_logical_joints_and_keeps_the_fault_snapshot():
    # 复现现场：只有 J1/J5/夹爪新鲜，四台达妙缺反馈；线槽位不等于关节号。
    ids = (1, 2, 8, 9, 0x14, 1, 2)
    raw = bytearray(feedback_bytes(ids=ids, fresh=False, flags=(0, 28, 0)))
    for joint_id in (1, 5, 7):
        raw[1 + 18 * p.SLOT_JOINT_IDS.index(joint_id) + 2] = 1
    link = UartLink(SimpleNamespace(), {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=ids)
    link._receive(raw)
    link._trip('post-enable feedback timeout')
    detail = dict(link.snapshot()['fault_detail'])
    assert detail['stale_logical_joints'] == [2, 3, 4, 6]
    assert detail['enabled_logical_joints'] == []
    assert detail['group_flags'] == [0, 28, 0]
    assert 'stale logical joints=[2, 3, 4, 6]' in link.fault
    link._receive(feedback_bytes(ids=ids, running=True))
    assert link.snapshot()['fault_detail'] == detail
    assert not link.enabled


def test_disconnect_after_enable_failure_preserves_the_original_fault():
    master, slave = pty.openpty()
    link = UartLink(PosixUart(os.ttyname(slave)), {i: Limits(-3., 3., 2., 10.) for i in range(1, 8)},
                    expected_motor_ids=(1, 2, 8, 9, 16, 1, 2))
    link._receive(feedback_bytes(fresh=False))
    link._trip('post-enable feedback timeout')
    original_fault = link.fault
    original_detail = dict(link.fault_detail)
    link.start()
    os.close(master)
    try:
        wait_for(lambda: link.stop_event.is_set())
        snapshot = link.snapshot()
        assert snapshot['fault'] == original_fault
        assert 'transport error' in snapshot['fault_detail']['transport_error_after_fault']
        assert all(snapshot['fault_detail'][key] == value for key, value in original_detail.items())
        assert not snapshot['enabled']
    finally:
        link.close()
        os.close(slave)
