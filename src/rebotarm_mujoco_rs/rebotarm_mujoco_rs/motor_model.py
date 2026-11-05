"""Simple motor model for the reBotArm simulation.

Implements a torque-speed envelope and a first-order thermal model so the
simulation can estimate whether a given motor can execute a motion and how
close it runs to its continuous (thermal) limit.

The torque-speed envelope is a constant-power derate above the rated speed:

    |v| <= v_rated          -> tau_max = peak
    v_rated < |v| < v_noload -> tau_max = peak * v_rated / |v|
    |v| >= v_noload         -> tau_max = 0

The thermal model tracks coil temperature from copper loss:

    P_copper = 1.5 * (|tau| / Kt)^2 * R_phase
    dT/dt    = (P_copper - (T - T_ambient) / Rth) / Cth

Rth / Cth are placeholders until calibrated against the real motors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointMotor:
    name: str
    motor: str
    rated: float
    peak: float
    mass: float
    gear: float
    torque_constant: float
    no_load_speed: float
    rated_speed: float
    rotor_inertia: float
    friction_static: float
    friction_viscous: float
    phase_resistance: float
    thermal_resistance: float
    thermal_capacity: float


class MotorModel:
    def __init__(self, joints: dict[str, JointMotor], ambient: float = 25.0) -> None:
        self._joints = dict(joints)
        self._ambient = float(ambient)
        self._temperature = {name: self._ambient for name in self._joints}

    @property
    def joint_names(self) -> list[str]:
        return list(self._joints)

    def spec(self, name: str) -> JointMotor:
        return self._joints[name]

    def reflected_inertia(self, name: str) -> float:
        spec = self._joints[name]
        return spec.gear * spec.gear * spec.rotor_inertia

    def torque_limit(self, name: str, speed: float) -> float:
        spec = self._joints[name]
        v = abs(float(speed))
        if v <= spec.rated_speed:
            return spec.peak
        if v >= spec.no_load_speed or spec.no_load_speed <= spec.rated_speed:
            return 0.0
        return spec.peak * spec.rated_speed / v

    def temperature(self, name: str) -> float:
        return self._temperature[name]

    def temperatures(self) -> dict[str, float]:
        return dict(self._temperature)

    def thermal_step(self, name: str, torque: float, dt: float) -> float:
        spec = self._joints[name]
        current = abs(float(torque)) / max(spec.torque_constant, 1e-6)
        copper_loss = 1.5 * current * current * spec.phase_resistance
        temp = self._temperature[name]
        cooling = (temp - self._ambient) / max(spec.thermal_resistance, 1e-6)
        temp += (copper_loss - cooling) / max(spec.thermal_capacity, 1e-6) * dt
        self._temperature[name] = temp
        return temp


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ValueError(f"motor_specs: missing {key!r} in {context}")
    return mapping[key]


def build_motor_model(specs: dict) -> MotorModel:
    motors = specs.get("motors") or {}
    joints: dict[str, JointMotor] = {}
    for name, joint in (specs.get("joints") or {}).items():
        motor_name = _require(joint, "motor", f"joints.{name}")
        motor = motors.get(motor_name)
        if motor is None:
            raise ValueError(f"motor_specs: motors.{motor_name} not defined")
        joints[name] = JointMotor(
            name=name,
            motor=str(motor_name),
            rated=float(_require(joint, "rated", f"joints.{name}")),
            peak=float(_require(joint, "peak", f"joints.{name}")),
            mass=float(motor.get("mass", 0.0)),
            gear=float(motor.get("gear", 1.0)),
            torque_constant=float(motor.get("torque_constant", 1.0)),
            no_load_speed=float(motor.get("no_load_speed", 0.0)),
            rated_speed=float(motor.get("rated_speed", 0.0)),
            rotor_inertia=float(motor.get("rotor_inertia", 0.0)),
            friction_static=float(motor.get("friction_static", 0.0)),
            friction_viscous=float(motor.get("friction_viscous", 0.0)),
            phase_resistance=float(motor.get("phase_resistance", 0.0)),
            thermal_resistance=float(motor.get("thermal_resistance", 1.0)),
            thermal_capacity=float(motor.get("thermal_capacity", 1.0)),
        )
    return MotorModel(joints, ambient=float(specs.get("ambient_temperature", 25.0)))


def load_motor_model(path: str | Path) -> MotorModel:
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return build_motor_model(yaml.safe_load(stream) or {})


__all__ = ["JointMotor", "MotorModel", "build_motor_model", "load_motor_model"]
