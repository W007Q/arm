"""Static gravity-torque scan for the reBotArm.

Sweeps the arm joint space, computes the joint torque required to hold each
pose statically (gravity only, qvel=0), and compares the per-joint maximum
against the configured motor rated/peak torque.

The result answers: "at the worst pose, can this motor hold the arm (and a
payload) statically, or does it need more torque / risk overheating?"

Run:
    ros2 run rebotarm_mujoco_rs rs_gravity_scan --payload 1.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rebotarm_description.parameters import description_path

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
GRAVITY = 9.81


def _default_motor_specs_path() -> Path:
    return description_path() / "config/motor_specs.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static gravity-torque scan.")
    parser.add_argument("--model", default="", help="MuJoCo model path (default: bundled scene).")
    parser.add_argument("--motor-specs", default="", help="motor_specs.yaml path.")
    parser.add_argument("--payload", type=float, default=None, help="Override payload mass (kg).")
    parser.add_argument("--samples", type=int, default=None, help="Random samples per scan.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-refine", action="store_true", help="Skip local refinement.")
    parser.add_argument("--output", default="", help="Write results CSV to this path.")
    parser.add_argument(
        "--pose",
        action="append",
        default=[],
        metavar="NAME=Q1,Q2,Q3,Q4,Q5,Q6",
        help="Report gravity torque at an exact joint pose; may be repeated.",
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Skip the global random/refined scan and only evaluate --pose entries.",
    )
    return parser


def _parse_named_pose(value: str, joint_count: int) -> tuple[str, np.ndarray]:
    name, separator, numbers = str(value).partition("=")
    if not separator or not name.strip():
        raise ValueError("pose must use NAME=Q1,Q2,Q3,Q4,Q5,Q6")
    q = np.asarray([float(item) for item in numbers.split(",")], dtype=float)
    if q.shape != (joint_count,) or not np.all(np.isfinite(q)):
        raise ValueError(f"pose {name!r} must contain {joint_count} finite values")
    return name.strip(), q


def _resolve_model_path(model_arg: str) -> str:
    if model_arg:
        return model_arg
    from .mujoco_sync import find_default_model

    return str(find_default_model())


class _Scanner:
    def __init__(self, model_path: str, specs: dict) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.arm_joints = [j for j in ARM_JOINTS if self._joint_id(j) >= 0]
        if not self.arm_joints:
            raise RuntimeError(f"no arm joints found in model: {model_path}")

        self.qpos_addr = np.array([self._qpos_addr(j) for j in self.arm_joints])
        self.dof_addr = np.array([self._dof_addr(j) for j in self.arm_joints])

        lo = self.model.jnt_range[[self._joint_id(j) for j in self.arm_joints], 0].copy()
        hi = self.model.jnt_range[[self._joint_id(j) for j in self.arm_joints], 1].copy()
        for name, bounds in (specs.get("limits") or {}).items():
            if name in self.arm_joints:
                idx = self.arm_joints.index(name)
                lo[idx], hi[idx] = float(bounds[0]), float(bounds[1])
        self.lo, self.hi = lo, hi

        payload = specs.get("payload") or {}
        tcp_body = str(payload.get("body") or "gripper_end")
        self.tcp_body_id = self._body_id(tcp_body)
        self.tcp_offset = np.asarray(payload.get('tcp_offset_m', [.0061, .1028, .1935]), dtype=float)
        if self.tcp_offset.shape != (3,) or not np.all(np.isfinite(self.tcp_offset)):
            raise ValueError('payload.tcp_offset_m must contain three finite values')
        self.masses = [float(m) for m in (payload.get("masses_kg") or [0.0])]

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def _joint_id(self, name: str) -> int:
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body_id(self, name: str) -> int:
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)

    def _qpos_addr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _dof_addr(self, name: str) -> int:
        return int(self.model.jnt_dofadr[self._joint_id(name)])

    def torque(self, q: np.ndarray, payload: float) -> np.ndarray:
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[self.qpos_addr] = q
        self.mujoco.mj_forward(self.model, d)
        tau = d.qfrc_bias[self.dof_addr].copy()
        if payload > 0.0 and self.tcp_body_id >= 0:
            point = d.xpos[self.tcp_body_id] + d.xmat[self.tcp_body_id].reshape(3, 3) @ self.tcp_offset
            self.mujoco.mj_jac(
                self.model, d, self.jacp, self.jacr, point, self.tcp_body_id
            )
            # Holding torque = -J^T f_ext, with f_ext = payload * gravity.
            # gravity is (0, 0, -g), so the payload adds +m*g*J_z to the
            # required joint torque.
            force = -payload * self.model.opt.gravity
            tau += self.jacp[:, self.dof_addr].T @ force
        return tau

    def scan(self, payload: float, samples: int, refine: bool, seed: int):
        rng = np.random.default_rng(seed)
        q_batch = rng.uniform(self.lo, self.hi, size=(samples, len(self.arm_joints)))
        max_abs = np.zeros(len(self.arm_joints))
        worst_q = np.zeros((len(self.arm_joints), len(self.arm_joints)))
        for q in q_batch:
            tau = self.torque(q, payload)
            for j in range(len(self.arm_joints)):
                value = abs(tau[j])
                if value > max_abs[j]:
                    max_abs[j] = value
                    worst_q[j] = q

        if not refine:
            return max_abs, worst_q

        # Robust global refinement of |tau_j| via differential evolution. The
        # objective tau_j**2 is smooth, so its maximum equals max|tau_j|.
        from scipy.optimize import differential_evolution

        bounds = list(zip(self.lo, self.hi))
        for j in range(len(self.arm_joints)):
            def objective(q, j=j):
                value = float(self.torque(q, payload)[j])
                return -(value * value)

            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=80,
                popsize=12,
                tol=1e-8,
                polish=True,
            )
            value = float(np.sqrt(max(-result.fun, 0.0)))
            if value > max_abs[j]:
                max_abs[j] = value
                worst_q[j] = result.x
        return max_abs, worst_q


def _verdict(value: float, rated: float, peak: float) -> str:
    if value <= rated:
        return "OK"
    if value <= peak:
        return "THERMAL_RISK"
    return "OVER_LIMIT"


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    specs_path = Path(args.motor_specs) if args.motor_specs else _default_motor_specs_path()
    specs = _load_yaml(specs_path)
    joint_specs = specs.get("joints") or {}
    scan_cfg = specs.get("scan") or {}

    model_path = _resolve_model_path(args.model)
    samples = int(args.samples if args.samples is not None else scan_cfg.get("samples", 20000))
    seed = int(args.seed if args.seed is not None else scan_cfg.get("seed", 0))
    refine = not args.no_refine and bool(scan_cfg.get("refine", True))

    scanner = _Scanner(model_path, specs)
    masses = [args.payload] if args.payload is not None else scanner.masses

    print(f"model        : {model_path}")
    print(f"motor specs  : {specs_path}")
    print(f"joints       : {scanner.arm_joints}")
    print(f"limits       : {dict(zip(scanner.arm_joints, zip(scanner.lo, scanner.hi)))}")
    print(f"tcp body     : {scanner.tcp_body_id}")
    print()

    rows = []
    named_poses = [_parse_named_pose(value, len(scanner.arm_joints)) for value in args.pose]
    for payload in masses:
        for pose_name, q in named_poses:
            if np.any(q < scanner.lo) or np.any(q > scanner.hi):
                raise ValueError(
                    f"pose {pose_name!r} is outside limits: {q.tolist()}"
                )
            tau = scanner.torque(q, payload)
            print(f"=== exact pose {pose_name!r}, payload {payload:.3f} kg ===")
            print(f"q = {np.array2string(q, precision=6, suppress_small=True)}")
            print(f"{'joint':8} {'motor':9} {'rated':>7} {'peak':>7} {'|tau|':>9} {'rated%':>8} {'verdict':>13}")
            for index, name in enumerate(scanner.arm_joints):
                spec = joint_specs.get(name) or {}
                rated = float(spec.get("rated", float("nan")))
                peak = float(spec.get("peak", float("nan")))
                value = abs(float(tau[index]))
                print(
                    f"{name:8} {str(spec.get('motor', '-')):9} {rated:7.2f} {peak:7.2f} "
                    f"{value:9.4f} {100.0 * value / rated:8.1f} {_verdict(value, rated, peak):>13}"
                )
            print()

    if args.pose_only:
        if not named_poses:
            raise ValueError("--pose-only requires at least one --pose")
        return 0

    for payload in masses:
        max_abs, worst_q = scanner.scan(payload, samples, refine, seed)
        print(f"=== payload {payload:.2f} kg ===")
        header = f"{'joint':8} {'motor':9} {'rated':>6} {'peak':>6} {'max|tau|':>9} {'verdict':>13}  worst pose"
        print(header)
        for j, name in enumerate(scanner.arm_joints):
            spec = joint_specs.get(name) or {}
            rated = float(spec.get("rated", float("nan")))
            peak = float(spec.get("peak", float("nan")))
            motor = str(spec.get("motor", "-"))
            value = float(max_abs[j])
            verdict = _verdict(value, rated, peak)
            pose = np.array2string(worst_q[j], precision=2, suppress_small=True)
            print(
                f"{name:8} {motor:9} {rated:6.1f} {peak:6.1f} {value:9.2f} "
                f"{verdict:>13}  {pose}"
            )
            rows.append(
                {
                    "payload_kg": payload,
                    "joint": name,
                    "motor": motor,
                    "rated_nm": rated,
                    "peak_nm": peak,
                    "max_abs_torque_nm": round(value, 4),
                    "verdict": verdict,
                    **{n: round(float(worst_q[j][k]), 4) for k, n in enumerate(scanner.arm_joints)},
                }
            )
        print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
