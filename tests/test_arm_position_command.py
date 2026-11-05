"""Contract tests for the single batch control interface.

These tests are pure Python and never open a serial port or enable hardware.
"""

import numpy as np
import pytest
import threading
from types import SimpleNamespace

from rebotarm_description.command_interface import validate_position_command


def test_batch_command_returns_independent_six_axis_copies():
    positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    limits = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    q, v = validate_position_command(positions, limits)
    assert q.shape == (6,)
    assert v.shape == (6,)
    np.testing.assert_allclose(q, positions)
    np.testing.assert_allclose(v, limits)
    q[0] = 9.0
    assert positions[0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "positions, limits, error",
    [
        ([0.0] * 5, [0.0] * 6, "each contain six"),
        ([0.0] * 6, [0.0] * 5, "each contain six"),
        ([float("nan")] + [0.0] * 5, [0.0] * 6, "finite"),
        ([0.0] * 6, [float("inf")] + [0.0] * 5, "finite"),
        ([0.0] * 6, [-0.1] + [0.0] * 5, "non-negative"),
    ],
)
def test_batch_command_rejects_malformed_values(positions, limits, error):
    with pytest.raises(ValueError, match=error):
        validate_position_command(positions, limits)


def adapters():
    from rebotarm_mujoco_rs.mujoco_sync import RsMujocoSync
    from rebotarm_uart.posvel_to_mit import PosVelToMit

    uart = object.__new__(PosVelToMit)
    uart.lock = threading.RLock()
    uart.names = [f"joint{i}" for i in range(1, 7)]
    uart.speed_limits = dict(zip(uart.names, [.3] * 6))
    uart.targets, uart.stamps = {}, {}
    uart.ever_commanded = False
    sim = object.__new__(RsMujocoSync)
    sim.arm_target_speed = np.full(6, .3)
    sim._clamp_arm_target = lambda q: q.copy()  # Targets below are inside URDF limits.
    sim.target_arm = np.zeros(6)
    sim._arm_command_speed = np.zeros(6)
    sim.last_input_time = 0.
    return uart, sim


def test_uart_and_mujoco_accept_same_absolute_target_and_zero_limit():
    uart, sim = adapters()
    msg = SimpleNamespace(positions=[.1, -.1, .2, -.2, .1, -.1],
                          velocity_limits=[0., .2, .4, .1, .2, .3])
    uart.on_arm_command(msg)
    sim._arm_command_callback(msg)
    np.testing.assert_allclose(sim.target_arm, [uart.targets[n][0] for n in uart.names])
    np.testing.assert_allclose(sim._arm_command_speed, [uart.targets[n][1] for n in uart.names])
    assert len(set(uart.stamps.values())) == 1
    # A zero limit pauses the reference even if the new goal differs.
    step = sim._slew_towards(np.zeros(6), sim.target_arm, sim._arm_command_speed, .02)
    assert step[0] == 0.
    assert step[1] < 0.


def test_invalid_batch_does_not_partially_change_either_adapter():
    uart, sim = adapters()
    valid = SimpleNamespace(positions=[.1]*6, velocity_limits=[.2]*6)
    uart.on_arm_command(valid)
    sim._arm_command_callback(valid)
    timestamps = dict(uart.stamps)
    bad = SimpleNamespace(positions=[.2]*6, velocity_limits=[.2]*5+[-1.])
    uart.on_arm_command(bad)
    sim._arm_command_callback(bad)
    assert uart.stamps == timestamps
    np.testing.assert_allclose(sim.target_arm, .1)
    assert all(value[0] == .1 for value in uart.targets.values())

