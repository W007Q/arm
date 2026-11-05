"""只用于 PTY 的 STM32 对端：收指令、周期反馈、500 ms 超时停止。"""
import os
import select
import struct
import threading
import time

from rebotarm_uart import protocol as p


def feedback_bytes(*, running=False, flags=None, ids=(1, 2, 8, 9, 16, 1, 2), stale_joint=None,
                   fault_joint=None, motorevo_status=None, fresh=True, positions=None):
    """独立按附件格式构造反馈，避免用待测解码器生成期望值。"""
    raw = bytearray(b'\xbb')
    if motorevo_status is None:
        motorevo_status = int(running)
    for i, motor_id in enumerate(ids, 1):
        status = 8 if i == fault_joint else int(running)
        temp = 30 + i if i <= 5 else 70 + i  # Motorevo 的温度编码加 40
        low, high = (35, 0) if i <= 5 else (motorevo_status & 255, motorevo_status >> 8)
        pos = 0. if positions is None else positions[i-1]
        raw.extend(struct.pack('<6Bfff', motor_id, status, int(fresh and i != stale_joint), temp,
                               low, high, pos, .01 * i, .02 * i))
    raw.extend(flags if flags is not None else ((3, 3, 3) if running else (0, 0, 0)))
    raw.append(0x44)
    return bytes(raw)


class Peer:
    def __init__(self, master, *, ids=(1, 2, 8, 9, 16, 1, 2), follow_commands=False,
                 positions=None):
        self.master = master
        self.decoder = p.Decoder(feedback=False)
        self.commands = []
        self.ids = ids
        self.follow_commands = follow_commands
        self.positions = [0.] * 7 if positions is None else list(positions)
        self.running = False
        self.feedback_enabled = True
        self.accept_enable = True
        self.stale_when_idle = False
        self.enable_delay = 0.
        self.flags = None
        self.stale_joint = None
        self.fault_joint = None
        self.motorevo_status = None
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        next_feedback = last_command = 0.
        enable_at = None
        try:
            while not self.stop.is_set():
                if select.select([self.master], [], [], .001)[0]:
                    for packet in self.decoder.feed(os.read(self.master, 8192)):
                        self.commands.append(packet)
                        if self.follow_commands:
                            targets = p.unpack_command(packet)
                            for i, joint_id in enumerate(p.SLOT_JOINT_IDS):
                                if targets[joint_id-1].vel > 0:
                                    self.positions[i] = targets[joint_id-1].pos
                        last_command = time.monotonic()
                        if enable_at is None:
                            enable_at = last_command + self.enable_delay
                now = time.monotonic()
                if now - last_command > .5:
                    enable_at = None
                self.running = self.accept_enable and enable_at is not None and now >= enable_at
                if self.feedback_enabled and now >= next_feedback:
                    raw = feedback_bytes(running=self.running, flags=self.flags,
                                         ids=self.ids, positions=self.positions,
                                         stale_joint=self.stale_joint, fault_joint=self.fault_joint,
                                         motorevo_status=self.motorevo_status,
                                         fresh=not self.stale_when_idle or self.running)
                    # 分片发送，模拟一次 read 不能取得完整帧。
                    for start in range(0, len(raw), 17):
                        os.write(self.master, raw[start:start + 17])
                    next_feedback = now + .02
        except OSError as exc:
            if not self.stop.is_set():
                self.error = exc

    def close(self):
        self.stop.set()
        self.thread.join(1.)
