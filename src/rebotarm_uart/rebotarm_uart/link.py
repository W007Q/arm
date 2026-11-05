"""七轴 UART 周期收发；故障时停发，由 STM32 看门狗执行停机。"""
from dataclasses import dataclass, replace
import math
import struct
import threading
import time

from . import protocol as p


@dataclass(frozen=True)
class Limits:
    """主机侧拒绝越限目标的安全边界，单位与协议（rad、rad/s、Nm）一致。"""

    lower: float
    upper: float
    velocity: float
    torque: float
    kp: float = 65.535
    kd: float = 5.

    def check(self, target):
        """拒绝越限目标；返回位置已对齐到合法 UART 刻度的目标。"""
        encoded_pos = p.command_values(target)[0]
        ranges = (('pos', target.pos, self.lower, self.upper, 'rad'),
                  ('vlim', target.vel, -self.velocity, self.velocity, 'rad/s'),
                  ('tau', target.tau, -self.torque, self.torque, 'Nm'),
                  ('kp', target.kp, 0., self.kp, ''), ('kd', target.kd, 0., self.kd, ''))
        for name, value, lower, upper, unit in ranges:
            if lower <= value <= upper:
                continue
            # CAN 的零位可能是 -0.000191 rad；仅当它按 UART 的
            # 0.001 rad 精度编码后落在限位内，才允许这个边界误差。
            if name == 'pos' and lower <= encoded_pos / 1000 <= upper:
                continue
            raise ValueError(f'joint {target.joint_id} exceeds configured limits: '
                             f'{name}={value:.9g} {unit}, allowed=[{lower:.9g}, {upper:.9g}] {unit}')
        # URDF 边界不一定是整毫弧度；向内取可发送刻度，避免四舍五入后越界。
        wire_lower, wire_upper = math.ceil(self.lower * 1000), math.floor(self.upper * 1000)
        if wire_lower > wire_upper:
            raise ValueError(f'joint {target.joint_id} limits contain no UART position step')
        return replace(target, pos=min(max(encoded_pos, wire_lower), wire_upper) / 1000)


@dataclass
class _EnableRequest:
    """首帧发出后的运行确认期限，不阻塞 enable 服务。"""

    deadline: float
    tx_frames_before: int


class UartLink:
    """七轴 72/131 字节协议的主机链路。

    附件协议没有独立 ARM/DISARM 帧：STM32 收到有效的 72 字节目标后按
    固件逻辑使能，停止接收目标后由 STM32 本地看门狗失能。因此 ROS 的
    ``enable`` 服务只负责打开主机的目标发送门，``disable`` 服务停止发
    送并由下位机看门狗退出控制。固定帧没有 CRC/序号，不能伪造 ACK；
    enable 首帧写入成功即返回；真实运行状态仍由 131 字节反馈确认。

    ``joint_id`` 是 ROS 逻辑关节号 1..7；发往 STM32 前按槽位映射重排。
    CAN 本地 ID 和反馈 ID 由 STM32 固件槽位映射。
    """

    def __init__(self, transport, limits, tx_hz=50., command_timeout=.12,
                 feedback_timeout=.25, start_tolerance=.15, *, expected_motor_ids,
                 slot_joint_ids=p.SLOT_JOINT_IDS,
                 clock=time.monotonic):
        expected_keys = tuple(sorted(limits))
        if expected_keys != tuple(range(1, p.FIXED_AXIS_COUNT + 1)):
            raise ValueError('fixed UART requires logical joint IDs 1..7')
        values = (tx_hz, command_timeout, feedback_timeout, start_tolerance)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('rates, timeouts and start tolerance must be positive')
        expected_motor_ids = tuple(expected_motor_ids)
        if len(expected_motor_ids) != p.FIXED_AXIS_COUNT or any(not 1 <= int(x) <= 255 for x in expected_motor_ids):
            raise ValueError('expected_motor_ids must contain seven valid CAN IDs')
        slot_joint_ids = p.validate_slot_joint_ids(slot_joint_ids)
        for lim in limits.values():
            if not all(math.isfinite(x) for x in (lim.lower, lim.upper, lim.velocity, lim.torque, lim.kp, lim.kd)):
                raise ValueError('invalid joint limits')
            if lim.lower >= lim.upper or min(lim.velocity, lim.torque, lim.kp, lim.kd) < 0:
                raise ValueError('invalid joint limits')
        self.transport, self.clock = transport, clock
        self.limits = {i: limits[i] for i in p.JOINT_IDS}
        self.tx_hz = float(tx_hz)
        self.command_timeout = float(command_timeout)
        self.feedback_timeout = float(feedback_timeout)
        self.start_tolerance = float(start_tolerance)
        self.expected_motor_ids = expected_motor_ids
        self.slot_joint_ids = slot_joint_ids
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.decoder = p.Decoder()
        self.enabled = False
        self.fault = ''
        self.fault_detail = {}
        self.feedback = None
        self.feedback_at = 0.
        self.feedback_generation = 0
        self.commands = {}
        self.pending = None               # 发包后异步确认七轴运行；服务不等待此结果
        self.tx_frames = self.rx_frames = self.bad_frames = self.missed_ticks = 0
        # 原始串口字节计数；用于区分“串口无数据”和“数据有但帧校验失败”。
        self.rx_bytes = 0
        # 保存最近一次 131 字节反馈被拒绝的原因，便于现场核对 ID/字段。
        self.last_bad_frame_error = ''

    def start(self):
        """启动固定帧收发线程；启动时保持停止发送。"""
        if self.thread is not None:
            raise RuntimeError('UART link already started')
        self.thread = threading.Thread(target=self._run, name='stm32-uart-fixed', daemon=True)
        self.thread.start()

    def set_targets(self, targets, complete=False):
        """缓存一组绝对目标；周期线程负责把它们编码为 72 字节。"""
        targets = tuple(targets)
        if len({t.joint_id for t in targets}) != len(targets):
            raise ValueError('duplicate target joint IDs')
        if complete and {t.joint_id for t in targets} != set(self.limits):
            raise ValueError('batch must contain all seven configured joints')
        checked = []
        for target in targets:
            if target.joint_id not in self.limits:
                raise ValueError('unknown logical joint ID')
            checked.append(self.limits[target.joint_id].check(target))
        now = self.clock()
        with self.lock:
            for target in checked:
                self.commands[target.joint_id] = (target, now)

    def _fresh_commands(self, now):
        return len(self.commands) == p.FIXED_AXIS_COUNT and all(
            now - stamp <= self.command_timeout for _, stamp in self.commands.values())

    def _trip(self, reason):
        """停止发包并锁定故障；真正失能由 STM32 看门狗执行。"""
        now = self.clock()
        # 保留触发瞬间的时间差；恢复后的 targets_fresh 不能解释历史故障。
        self.fault_detail = dict(
            command_age_sec={str(key): now-stamp for key, (_, stamp) in self.commands.items()},
            feedback_age_sec=now-self.feedback_at if self.feedback is not None else None,
            command_timeout_sec=self.command_timeout,
            tx_frames=self.tx_frames,
            feedback_generation=self.feedback_generation,
        )
        if self.feedback is not None:
            # 锁存触发时的逐轴状态，避免后续失能反馈覆盖使能失败的线索。
            self.fault_detail.update(
                stale_logical_joints=sorted(j.joint_id for j in self.feedback.joints if not j.feedback_fresh),
                enabled_logical_joints=sorted(j.joint_id for j in self.feedback.joints if j.enabled),
                group_flags=list(self.feedback.group_flags),
            )
        if reason == 'post-enable feedback timeout':
            if self.feedback is None or now-self.feedback_at > self.feedback_timeout:
                reason += '; no recent 131-byte UART report'
            else:
                reason += (f"; stale logical joints={self.fault_detail['stale_logical_joints']}"
                           f"; enabled logical joints={self.fault_detail['enabled_logical_joints']}"
                           f"; group_flags={self.fault_detail['group_flags']}")
        self.enabled = False
        self.fault = str(reason)
        self.commands.clear()
        self.pending = None

    def request(self, enable, timeout=2.):
        """处理 ROS enable/disable 服务。

        固定协议没有 ARM/DISARM 线帧。首次发包不要求 CAN 反馈新鲜，
        首帧写入成功即返回，不读取反馈判断本次服务是否成功。
        后续异步检查七轴运行反馈，确认前只持续发送零输出帧。
        disable 确认主机停发，实际停机依赖 STM32 看门狗。
        """
        with self.lock:
            if not enable:
                self.pending = None
                self.enabled = False
                self.commands.clear()
                self.fault = ''
                self.fault_detail = {}
                return True, 'command stream stopped; STM32 watchdog must stop motors'
            now = self.clock()
            if self.stop_event.is_set():
                return False, 'UART worker stopped'
            if self.enabled:
                return True, 'already enabled'
            if self.pending is not None:
                return True, 'enable command already sent; motor feedback check pending'
            if self.fault:
                return False, f'latched fault: {self.fault}; disable explicitly before re-enabling'
            self.pending = _EnableRequest(now + timeout, self.tx_frames)
            try:
                # 直接写首帧；不等待 UART/CAN 反馈，也不读取上游运动目标。
                self._send_command(self._zero_output())
            except (OSError, TimeoutError) as exc:
                self._trip(f'UART transport error: {exc}')
                self.stop_event.set()
                return False, self.fault
            return True, 'enable command sent; motor feedback will be checked asynchronously'

    def _zero_output(self):
        """冻结位置轨迹、增益和前馈清零，供首次发包及后续确认阶段使用。"""
        return [p.MitTarget(key, 0., 0., 0., 0., 0.) for key in self.limits]

    def _send_command(self, targets):
        packet = p.pack_command(targets, self.slot_joint_ids)
        self.transport.write(packet)
        self.tx_frames += 1

    def _receive(self, packet):
        feedback = p.unpack_feedback(packet, self.expected_motor_ids, self.slot_joint_ids)
        now = self.clock()
        self.feedback = feedback
        self.feedback_at = now
        self.feedback_generation += 1
        if feedback.fault or feedback.state == 3:
            self._trip(f'STM32 fixed feedback fault 0x{feedback.fault:06x}')
        elif self.enabled and any(not joint.feedback_fresh for joint in feedback.joints):
            self._trip('STM32 reports stale motor feedback')
        elif (self.pending is not None and self.tx_frames > self.pending.tx_frames_before
              and feedback.state == 1 and feedback.command_fresh
              and all(joint.feedback_fresh for joint in feedback.joints)):
            actual = {joint.joint_id: joint.pos for joint in feedback.joints}
            if self._fresh_commands(now):
                # 只有拿到新测量后，才校验正常目标与实际位置的起始偏差。
                if any(abs(self.commands[key][0].pos - actual[key]) > self.start_tolerance for key in self.limits):
                    self._trip('initial targets too far from measured positions')
                    return
            else:
                # 适配器也在等新测量。短暂保持零输出，留出正常目标接管的时间。
                holds = [p.MitTarget(key, actual[key], 0., 0., 0., 0.) for key in self.limits]
                try:
                    self.set_targets(holds, complete=True)
                except ValueError as exc:
                    self._trip(str(exc))
                    return
            self.enabled = True
            self.pending = None
        elif self.enabled and (feedback.state != 1 or not feedback.command_fresh):
            self._trip('STM32 reports stopped motor group or stale command')

    def _tick(self, now):
        # pending enable 也必须发送目标，STM32 没有单独 ARM 帧。
        if not self.enabled and self.pending is None:
            return
        if self.pending is not None:
            if now > self.pending.deadline:
                self._trip('post-enable feedback timeout')
                return
            # 速度为 0 会冻结 STM32 的位置轨迹；Kp/Kd/tau 为 0，
            # 不用缓存位置施加控制力矩，也不提前执行任务目标或重力前馈。
            targets = self._zero_output()
        else:
            if self.feedback is None or now - self.feedback_at > self.feedback_timeout:
                self._trip('UART feedback timeout')
                return
            if not self._fresh_commands(now):
                self._trip('upstream joint command timeout')
                return
            targets = [self.commands[key][0] for key in self.limits]
        self._send_command(targets)

    def snapshot(self):
        with self.lock:
            now = self.clock()
            ready = self.feedback is not None and now - self.feedback_at <= self.feedback_timeout
            state = ('FAULT' if self.fault else 'ENABLED' if self.enabled else 'ENABLING' if self.pending
                     else 'DISABLED' if ready else 'CONNECTING')
            command_snapshot = {
                str(key): {
                    'pos': target.pos, 'vel': target.vel, 'kp': target.kp,
                    'kd': target.kd, 'tau': target.tau,
                    'age_sec': now - stamp,
                }
                for key, (target, stamp) in self.commands.items()
            }
            return dict(state=state, enabled=self.enabled,
                        command_stream_active=self.enabled or self.pending is not None,
                        feedback_ready=ready,
                        fault=self.fault, fault_detail=self.fault_detail, feedback=self.feedback,
                        feedback_age=now - self.feedback_at if self.feedback else None,
                        feedback_generation=self.feedback_generation,
                        tx_frames=self.tx_frames, rx_frames=self.rx_frames,
                        rx_bytes=self.rx_bytes,
                        decoder_buffered_bytes=len(self.decoder.buffer),
                        bad_frames=self.bad_frames, missed_ticks=self.missed_ticks,
                        targets_fresh=self._fresh_commands(now),
                        command_snapshot=command_snapshot,
                        last_bad_frame_error=self.last_bad_frame_error)

    def _run(self):
        next_tick = self.clock()
        last_byte = self.clock()
        try:
            while not self.stop_event.is_set():
                now = self.clock()
                data = self.transport.read()
                with self.lock:
                    if now - last_byte > .05:
                        self.decoder.reset_after_timeout()
                    if data:
                        last_byte = now
                        self.rx_bytes += len(data)
                        for packet in self.decoder.feed(data):
                            try:
                                self._receive(packet)
                                self.rx_frames += 1
                            except (ValueError, struct.error) as exc:
                                self.bad_frames += 1
                                self.last_bad_frame_error = str(exc)
                    if now >= next_tick:
                        self._tick(now)
                        period = 1. / self.tx_hz
                        if now - next_tick >= period:
                            self.missed_ticks += int((now - next_tick) / period)
                        next_tick = now + period
                self.stop_event.wait(min(.001, max(0., next_tick - self.clock())))
        except (OSError, TimeoutError, ValueError) as exc:
            with self.lock:
                self.enabled = False
                error = f'UART transport error: {exc}'
                if self.fault:
                    # 排查时常会先断电；保留最初的超时/电机故障，另记断线。
                    self.fault_detail['transport_error_after_fault'] = error
                else:
                    self._trip(error)
                self.pending = None
                self.stop_event.set()
        finally:
            with self.lock:
                self.enabled = False
                self.pending = None

    def close(self):
        """停止主机发包并关闭串口；STM32 由看门狗完成最终失能。"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        self.transport.close()
