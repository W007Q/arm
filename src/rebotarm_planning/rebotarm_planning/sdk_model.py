"""Hardware-free robot mathematics (Pinocchio FK/IK/gravity).

Goal: make the simulation use the SAME kinematics/dynamics as the real
reBotArm driver (``reBotArm_control_py``), so joint targets planned in the
simulation transfer directly to the hardware / STM32.

This module is intentionally self-contained and hardware-free: it only loads a
URDF and computes FK, IK and gravity. The MuJoCo model is still used by the
simulation as the physics engine.

The third-party SDK is expected at ``third_party/reBotArm_control_py`` and is
added to ``sys.path`` on demand. ``pinocchio`` and ``motorbridge`` are runtime
dependencies (``pip install --user pin motorbridge`` or the ROS pinocchio apt
package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# Default SDK layout relative to the workspace root.
_SDK_RELATIVE = Path("third_party") / "reBotArm_control_py"
_DEFAULT_URDF = Path("src/rebotarm_description/urdf/robot_arm.urdf")
_DEFAULT_FRAME = "gripper_end"


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _SDK_RELATIVE).is_dir():
            return parent
    # fall back to the repo root inferred from the source tree
    return here.parents[3]


def ensure_sdk_on_path() -> Path:
    """Add the third-party SDK to ``sys.path`` and return its root."""
    root = _workspace_root() / _SDK_RELATIVE
    if not (root / "reBotArm_control_py").is_dir():
        raise FileNotFoundError(
            f"reBotArm_control_py not found at {root}. Clone it first:\n"
            "  git clone https://github.com/vectorBH6/reBotArm_control_py.git "
            "third_party/reBotArm_control_py"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_urdf_path() -> Path:
    return _workspace_root() / _DEFAULT_URDF


class SdkRobotModel:
    """Thin wrapper over the SDK's pinocchio FK / CLIK IK / gravity.

    All methods take/return ``controlled_joints`` values (the 6 arm joints);
    the URDF may contain extra passive joints (e.g. the two gripper fingers),
    which are padded with their neutral value.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        end_effector_frame: str = _DEFAULT_FRAME,
        tcp_offset=None,
        controlled_joints: int = 6,
    ) -> None:
        ensure_sdk_on_path()
        import pinocchio as pin
        from reBotArm_control_py.kinematics import (
            compute_fk as sdk_compute_fk,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import (
            IKParams,
            solve_ik_with_retry,
        )
        from reBotArm_control_py.dynamics import compute_generalized_gravity

        self._pin = pin
        self._sdk_compute_fk = sdk_compute_fk
        self._solve_ik_with_retry = solve_ik_with_retry
        self._ik_params = IKParams
        self._gravity_fn = compute_generalized_gravity

        path = Path(urdf_path) if urdf_path else default_urdf_path()
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")
        self.urdf_path = path
        if str(path).endswith('.xacro'):
            import xacro
            self.model = pin.buildModelFromXML(xacro.process_file(str(path)).toxml())
        else:
            self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()
        self.nq = int(self.model.nq)
        self.n_ctrl = min(int(controlled_joints), self.nq)

        frame_id = self.model.getFrameId(end_effector_frame)
        if frame_id >= len(self.model.frames) or frame_id < 0:
            raise ValueError(f"end-effector frame {end_effector_frame!r} not in URDF")
        self.end_effector_frame = end_effector_frame
        self.frame_id = int(frame_id)
        self.tcp_offset = (
            np.asarray(tcp_offset, dtype=np.float64)
            if tcp_offset is not None
            else np.zeros(3)
        )
        if self.tcp_offset.shape != (3,):
            raise ValueError("tcp_offset must have 3 elements")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pad(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        padded = np.zeros(self.nq)
        n = min(q.shape[0], self.n_ctrl)
        padded[:n] = q[:n]
        return padded

    def joint_limits(self) -> list[tuple[float, float]]:
        limits = []
        for j in range(self.n_ctrl):
            lo = float(self.model.lowerPositionLimit[j])
            hi = float(self.model.upperPositionLimit[j])
            limits.append((lo, hi))
        return limits

    # ------------------------------------------------------------------
    # forward / inverse kinematics, gravity
    # ------------------------------------------------------------------

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (with offset) and rotation for joint positions q."""
        pos, rot, _ = self._sdk_compute_fk(
            self.model, self._pad(q), frame_name=self.end_effector_frame
        )
        pos = pos + rot @ self.tcp_offset
        return np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)

    def ik(
        self,
        target_pos,
        target_rot,
        q_seed=None,
        max_retries: int = 8,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve IK for a TCP pose. Returns (q, success, error)."""
        pin = self._pin
        rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
        # The target is for the TCP; shift it back to the end-effector frame.
        ee_pos = np.asarray(target_pos, dtype=np.float64).reshape(3) - rot @ self.tcp_offset
        target = pin.SE3(rot, ee_pos)
        seed = self._pad(q_seed) if q_seed is not None else np.zeros(self.nq)
        result = self._solve_ik_with_retry(
            self.model,
            self.data,
            self.frame_id,
            target,
            seed,
            max_retries=max_retries,
        )
        return np.asarray(result.q[: self.n_ctrl], dtype=np.float64), bool(
            result.success
        ), float(result.error)

    def gravity(self, q) -> np.ndarray:
        """Generalized gravity torque g(q) for the controlled joints."""
        tau = self._gravity_fn(self.model, self._pad(q), self.data)
        return np.asarray(tau[: self.n_ctrl], dtype=np.float64)
