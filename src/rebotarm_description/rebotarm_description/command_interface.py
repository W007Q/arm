"""Validation shared by the batch arm command adapters.

The ROS message is intentionally small, but every backend must apply the same
shape and finite-value checks before touching a controller.  Keeping this
check in the description package avoids each UART, MuJoCo and SDK adapter
silently accepting a different command contract.
"""

from __future__ import annotations

import numpy as np


ARM_POSITION_COUNT = 6


def validate_position_command(positions, velocity_limits) -> tuple[np.ndarray, np.ndarray]:
    """Return validated copies of a six-axis absolute target and its limits.

    ``velocity_limits`` is a non-negative *position trajectory limit* used by
    the STM32's interpolator.  Zero is valid and pauses that axis; it is not a
    signed target velocity.
    """

    try:
        q = np.asarray(positions, dtype=float)
        v = np.asarray(velocity_limits, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions and velocity_limits must be numeric arrays") from exc
    if q.shape != (ARM_POSITION_COUNT,) or v.shape != (ARM_POSITION_COUNT,):
        raise ValueError("positions and velocity_limits must each contain six values")
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("positions and velocity_limits must be finite")
    if np.any(v < 0.0):
        raise ValueError("velocity_limits must be non-negative")
    return q.copy(), v.copy()
