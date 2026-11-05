"""STM32 七轴固定协议：发送 72 字节，接收 131 字节，均为小端。

位置是相对 MCU 启动零点的绝对 rad；速度字段为限速值（0 暂停位置轨迹）。
没有 CRC、序号和独立使能帧。
"""
from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class MitTarget:
    """一轴 MIT 目标；joint_id 是 ROS 逻辑关节号 1..7。

    UART 帧会按 ``SLOT_JOINT_IDS`` 把逻辑关节重排为 STM32 槽位顺序。
    角度/限速/力矩单位分别为 rad、rad/s、Nm；固定帧用 1000 倍整数传输。
    保留 vel 字段名以对应 ROS 消息；固件使用其绝对值，方向由位置差决定。
    """

    joint_id: int
    pos: float
    vel: float
    kp: float
    kd: float
    tau: float


@dataclass(frozen=True)
class MotorState:
    """STM32 经 CAN 汇总后返回的一轴实测状态。"""

    joint_id: int
    status: int
    pos: float
    vel: float
    tau: float
    temperature: float
    # 本地 Motor ID 可跨 CAN 总线重复；逻辑关节始终按固定槽位映射。
    motor_id: int = 0
    feedback_fresh: bool = True
    mos_temperature: float = math.nan
    rotor_temperature: float = math.nan
    enabled: bool = False


@dataclass(frozen=True)
class Feedback:
    """一整组状态反馈。

    三组 flags 分别保留；state 是主机推导值，不是线上的状态字段。
    """

    state: int
    fault: int
    joints: tuple
    # 当前固件的三个处理组标志；组名不等于实体关节电机型号。
    group_flags: tuple  # STM32 J4310、J4340P、Motorevo 各一字节

    @property
    def command_fresh(self):
        return all(flag & FEEDBACK_FLAG_COMMAND_FRESH for flag in self.group_flags)


COMMAND_START = 0xAA
COMMAND_END = 0x55
FEEDBACK_START = 0xBB
FEEDBACK_END = 0x44
COMMAND_SIZE = 72
FEEDBACK_SIZE = 131
FIXED_AXIS_COUNT = 7

# ROS 逻辑关节号始终为 1..7；STM32 槽位顺序由下方映射指定。
JOINT_IDS = tuple(range(1, 8))
# 固件槽位 1..7 -> ROS 逻辑关节：J4、J6、J1、J2、J3、J5、J7。
SLOT_JOINT_IDS = (4, 6, 1, 2, 3, 5, 7)
FEEDBACK_FLAG_COMMAND_FRESH = 1 << 0
FEEDBACK_FLAG_RUNNING = 1 << 1

# 每轴 10 字节：位置、速度、Kp、Kd、前馈力矩，所有物理量除以 1000。
FIXED_COMMAND_JOINT = struct.Struct('<hhHHh')
# 每轴 18 字节：ID、状态/使能、反馈新鲜、温度/状态扩展和 3 个 float32。
FIXED_FEEDBACK_JOINT = struct.Struct('<6Bfff')


def _quantize(value, lower, upper, name):
    """把有限浮点数按固定缩放量化，并拒绝溢出。"""
    if not math.isfinite(value) or not lower / 1000 <= value <= upper / 1000:
        raise ValueError(f'{name}={value} is outside fixed-frame range')
    return int(round(value * 1000))


def _quantize_gain(value, upper, name):
    """增益按固件参考工具的正数四舍五入规则量化。"""
    if not math.isfinite(value) or not 0 <= value <= upper / 1000:
        raise ValueError(f'{name}={value} is outside fixed-frame range')
    return int(math.floor(value * 1000 + 0.5))


def command_values(target):
    """缓存和编码共用一处检查：固件接收 Kp≤65.535、Kd≤5。"""
    return (
        _quantize(target.pos, -32768, 32767, 'absolute position'),
        _quantize(target.vel, -32768, 32767, 'velocity'),
        _quantize_gain(target.kp, 65535, 'Kp'),
        _quantize_gain(target.kd, 5000, 'Kd'),
        _quantize(target.tau, -32768, 32767, 'torque'),
    )


def validate_slot_joint_ids(slot_joint_ids):
    """检查并标准化“线槽位 -> 逻辑关节号”排列。"""
    values = tuple(int(value) for value in slot_joint_ids)
    if len(values) != FIXED_AXIS_COUNT or sorted(values) != list(JOINT_IDS):
        raise ValueError('slot_joint_ids must be a permutation of logical joints 1..7')
    return values


def pack_command(targets, slot_joint_ids=SLOT_JOINT_IDS):
    """编码七轴绝对位置，并按 STM32 槽位重排；帧内没有看门狗参数。"""
    targets = tuple(targets)
    if tuple(t.joint_id for t in targets) != JOINT_IDS:
        raise ValueError('command requires logical joint order 1..7')
    slot_joint_ids = validate_slot_joint_ids(slot_joint_ids)
    by_joint = {target.joint_id: target for target in targets}
    output = bytearray([COMMAND_START])
    for joint_id in slot_joint_ids:
        target = by_joint[joint_id]
        output.extend(FIXED_COMMAND_JOINT.pack(*command_values(target)))
    output.append(COMMAND_END)
    assert len(output) == COMMAND_SIZE
    return bytes(output)


def unpack_command(packet, slot_joint_ids=SLOT_JOINT_IDS):
    """解析 72 字节指令；返回逻辑关节 1..7 的绝对目标。"""
    packet = bytes(packet)
    if len(packet) != COMMAND_SIZE or packet[0] != COMMAND_START or packet[-1] != COMMAND_END:
        raise ValueError('invalid fixed command frame')
    slot_joint_ids = validate_slot_joint_ids(slot_joint_ids)
    targets = {}
    for index in range(FIXED_AXIS_COUNT):
        offset = 1 + index * FIXED_COMMAND_JOINT.size
        values = FIXED_COMMAND_JOINT.unpack_from(packet, offset)
        targets[slot_joint_ids[index]] = MitTarget(slot_joint_ids[index], *(v / 1000 for v in values))
    return tuple(targets[joint_id] for joint_id in JOINT_IDS)


def unpack_feedback(packet, expected_motor_ids=None, slot_joint_ids=SLOT_JOINT_IDS):
    """解析 131 字节反馈，并把 STM32 槽位映射到逻辑关节 1..7。

    expected_motor_ids 来自硬件配置；离线解析可省略，运行链路必须传入。
    """
    packet = bytes(packet)
    if len(packet) != FEEDBACK_SIZE or packet[0] != FEEDBACK_START or packet[-1] != FEEDBACK_END:
        raise ValueError('invalid fixed feedback frame')
    expected = tuple(expected_motor_ids) if expected_motor_ids is not None else None
    if expected is not None and len(expected) != FIXED_AXIS_COUNT:
        raise ValueError('expected_motor_ids must contain seven IDs')
    slot_joint_ids = validate_slot_joint_ids(slot_joint_ids)
    joints = []
    for index in range(FIXED_AXIS_COUNT):
        offset = 1 + index * FIXED_FEEDBACK_JOINT.size
        motor_id, status_byte, fresh, temp_code, status_low, status_high, pos, vel, tau = FIXED_FEEDBACK_JOINT.unpack_from(packet, offset)
        if motor_id == 0 or (expected is not None and motor_id != expected[index]):
            expected_text = '' if expected is None else f', expected {expected[index]:#x}'
            raise ValueError(f'unexpected motor ID at feedback slot {index + 1}: '
                             f'got {motor_id:#x}{expected_text}')
        if fresh not in (0, 1):
            raise ValueError(f'invalid feedback freshness flag at slot {index + 1}')
        if not all(math.isfinite(v) for v in (pos, vel, tau)):
            raise ValueError(f'non-finite feedback at slot {index + 1}')
        if (index >= 5 and status_byte not in (0, 1)) or (index < 5 and status_byte not in (0, 1) and not 8 <= status_byte <= 14):
            raise ValueError(f'invalid motor status at slot {index + 1}')
        status = status_low | (status_high << 8) if index >= 5 else status_byte
        temperature = float(temp_code - 40) if index >= 5 else float(temp_code)
        joints.append(MotorState(slot_joint_ids[index], status, pos, vel, tau, temperature,
                                 motor_id=motor_id, feedback_fresh=bool(fresh),
                                 mos_temperature=float(temp_code) if index < 5 else math.nan,
                                 rotor_temperature=float(status_low) if index < 5 else math.nan,
                                 enabled=status_byte == 1))
    group_flags = tuple(packet[127:130])
    # 与固件一致：达妙使用 8..E 故障码；Motorevo 除 bit0 外均为故障位。
    fault = sum(1 << index for index, joint in enumerate(joints)
                if (index < 5 and 8 <= joint.status <= 14) or (index >= 5 and joint.status & 0xfffe))
    running = all(flag & FEEDBACK_FLAG_RUNNING for flag in group_flags) and all(j.enabled for j in joints)
    disabled = not any(flag & FEEDBACK_FLAG_RUNNING for flag in group_flags) and not any(j.enabled for j in joints)
    state = 3 if fault else 1 if running else 0 if disabled else 2  # 2 表示部分轴运行
    return Feedback(state, fault, tuple(joints), group_flags)


class Decoder:
    """固定帧分片、粘包和噪声处理；默认接收 131 字节反馈。"""

    def __init__(self, *, feedback=True):
        self.buffer = bytearray()
        self.start, self.end, self.size = ((FEEDBACK_START, FEEDBACK_END, FEEDBACK_SIZE)
                                         if feedback else (COMMAND_START, COMMAND_END, COMMAND_SIZE))

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while self.buffer:
            if self.buffer[0] != self.start:
                del self.buffer[0]
                continue
            if len(self.buffer) < self.size:
                break
            if self.buffer[self.size - 1] != self.end:
                del self.buffer[0]
                continue
            packets.append(bytes(self.buffer[:self.size]))
            del self.buffer[:self.size]
        return packets

    def reset_after_timeout(self):
        self.buffer.clear()
