"""固定线格式检查；期望字节直接按 STM32 附件布局构造。"""
import math
import struct

import pytest

from rebotarm_uart import protocol as p


def targets():
    return tuple(p.MitTarget(i, .1, -.02, 1.2, .6, -.03) for i in range(1, 8))


FIRMWARE_IDS = (1, 2, 8, 9, 16, 1, 2)
SLOT_JOINT_IDS = (4, 6, 1, 2, 3, 5, 7)


def feedback(flags=(3, 3, 3), ids=FIRMWARE_IDS, motorevo_status=1):
    raw = bytearray(b'\xbb')
    for i, motor_id in enumerate(ids):
        raw.extend(struct.pack('<6Bfff', motor_id, 1, 1, 30 if i < 5 else 70,
                               35 if i < 5 else (motorevo_status & 255),
                               0 if i < 5 else (motorevo_status >> 8), .1, .2, .3))
    return bytes(raw) + bytes(flags) + b'\x44'


def test_command_matches_golden_bytes_and_absolute_position():
    packet = p.pack_command(targets())
    assert packet == b'\xaa' + bytes.fromhex('6400ecffb0045802e2ff') * 7 + b'\x55'
    assert len(packet) == 72
    assert p.unpack_command(packet) == targets()
    assert p.pack_command(targets()) == packet  # 重发同一目标不会累加位置


def test_command_and_feedback_remap_wire_slots_to_logical_joints():
    logical = tuple(p.MitTarget(i, float(i), .02, 1.2, .6, -.03) for i in range(1, 8))
    packet = p.pack_command(logical, SLOT_JOINT_IDS)
    decoded = p.unpack_command(packet, SLOT_JOINT_IDS)
    assert [target.pos for target in decoded] == list(map(float, range(1, 8)))
    # The first wire block is J4 and the third is J1.
    assert struct.unpack_from('<h', packet, 1)[0] == 4000
    assert struct.unpack_from('<h', packet, 21)[0] == 1000
    decoded_feedback = p.unpack_feedback(feedback(), FIRMWARE_IDS, SLOT_JOINT_IDS)
    assert [joint.joint_id for joint in decoded_feedback.joints] == list(SLOT_JOINT_IDS)


@pytest.mark.parametrize('field,value', [('kp', 65.536), ('kp', -.001), ('kd', 5.001),
                                         ('pos', 32.768), ('tau', -32.769), ('vel', math.nan)])
def test_command_rejects_wire_overflow(field, value):
    from dataclasses import replace
    bad = list(targets())
    bad[0] = replace(bad[0], **{field: value})
    with pytest.raises(ValueError):
        p.pack_command(bad)


def test_command_requires_seven_slots_in_order():
    for bad in (targets()[:-1], targets()[::-1]):
        with pytest.raises(ValueError):
            p.pack_command(bad)


def test_feedback_keeps_temperatures_and_motorevo_status_separate():
    decoded = p.unpack_feedback(feedback(), FIRMWARE_IDS)
    assert decoded.state == 1 and decoded.command_fresh
    assert decoded.group_flags == (3, 3, 3)
    assert decoded.fault == 0
    assert decoded.joints[0].mos_temperature == 30
    assert decoded.joints[0].rotor_temperature == 35
    assert decoded.joints[5].temperature == 30
    assert math.isnan(decoded.joints[5].mos_temperature)
    assert decoded.joints[5].status == 1
    assert decoded.joints[5].enabled


@pytest.mark.parametrize('status', [2, 0x4000, 0x8000, 0xc001])
def test_motorevo_error_bits_follow_the_flashed_firmware(status):
    decoded = p.unpack_feedback(feedback(motorevo_status=status), FIRMWARE_IDS)
    assert decoded.state == 3 and decoded.fault == 0x60


def test_kd_firmware_boundary_is_accepted():
    from dataclasses import replace
    command = [replace(t, kd=5.) for t in targets()]
    assert p.unpack_command(p.pack_command(command)) == tuple(command)


@pytest.mark.parametrize('flags', [(0, 3, 3), (3, 0, 3), (3, 3, 0)])
def test_running_requires_all_three_groups(flags):
    decoded = p.unpack_feedback(feedback(flags), FIRMWARE_IDS)
    assert decoded.state != 1
    assert not decoded.command_fresh


def test_feedback_can_validate_local_can_ids_when_configured():
    ids = FIRMWARE_IDS
    raw = feedback(ids=ids)
    with pytest.raises(ValueError):
        p.unpack_feedback(raw, range(1, 8))
    assert [j.motor_id for j in p.unpack_feedback(raw, ids).joints] == list(ids)


@pytest.mark.parametrize('offset,data', [(1, b'\x07'), (3, b'\x02'), (2, b'\x07'),
                                        (7, bytes.fromhex('0000c07f'))])
def test_invalid_feedback_is_rejected(offset, data):
    raw = bytearray(feedback())
    raw[offset:offset + len(data)] = data
    with pytest.raises(ValueError):
        p.unpack_feedback(raw, FIRMWARE_IDS)


def test_decoder_recovers_from_noise_fragments_and_bad_tail():
    raw = feedback()
    bad = raw[:-1] + b'\x00'
    decoder = p.Decoder()
    output = []
    for byte in b'noise' + bad + raw + raw:
        output.extend(decoder.feed(bytes([byte])))
    assert output == [raw, raw]
    assert p.Decoder().feed(raw + raw) == [raw, raw]
    assert decoder.feed(raw[:15]) == []
    decoder.reset_after_timeout()
    assert decoder.feed(raw) == [raw]
