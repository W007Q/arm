"""Exercise the actual ROS plant physics loop without starting ROS or any motor."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rebotarm_mujoco_rs.mujoco_sync import RsMujocoSync

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gravity_bench', ROOT/'scripts/validate_gravity_compensation.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
GRASP = np.array([-.335855,-1.914513,1.684191,-1.343377,.001031,-1.911958])


def plant(scale):
    model, motor = bench.physics_model()
    data = mujoco.MjData(model)
    pos, dofs = bench.addresses(model)
    data.qpos[pos] = GRASP
    mujoco.mj_forward(model, data)
    master = model.joint('joint7')
    return SimpleNamespace(
        model=model, data=data, motor_model=motor, update_rate=1000.,
        compliance_enabled=False, compliance_scale=1., pingpong_enabled=False,
        arm_kp=np.full(6,5.), arm_kd=np.full(6,2.), gravity_bias_scale=scale,
        arm_tau_limit=bench.FIXED_LIMITS, arm_qpos_addrs=pos, arm_dof_addrs=dofs,
        arm_actuator_ids=np.array([model.actuator(n+'_motor').id for n in bench.ARM_JOINTS]),
        target_arm=GRASP.copy(), _applied_arm_target=GRASP.copy(), _arm_command_speed=np.ones(6),
        target_gripper=0., _applied_gripper_target=0., gripper_target_speed=.02,
        control_mode='joint', cartesian_target=None,
        master_qpos_addr=int(master.qposadr[0]), master_dof_addr=int(master.dofadr[0]),
        master_actuator_id=model.actuator('joint7_motor').id,
        gripper_kp=80., gripper_kd=10., gripper_tau_limit=150., _command_gripper_force_limit=0.,
        _slew_towards=RsMujocoSync._slew_towards,
    )


@pytest.mark.parametrize('scale', [0., .3, 1.])
def test_actual_physics_loop_scales_bias_and_keeps_motor_caps(scale):
    p = plant(scale)
    gravity = p.data.qfrc_bias[p.arm_dof_addrs].copy()
    RsMujocoSync._update_physics(p)
    np.testing.assert_allclose(p._last_arm_tau, scale*gravity, atol=1e-10)
    assert np.all(np.abs(p._last_arm_tau) <= [27.,27.,27.,7.,13.,7.])


def test_actual_plant_full_bias_holds_but_no_bias_sags():
    results = []
    for scale in (0., 1.):
        p = plant(scale)
        p.update_rate = 250.
        for _ in range(1500):
            RsMujocoSync._update_physics(p)
        results.append(float(np.max(np.abs(np.rad2deg(p.data.qpos[p.arm_qpos_addrs]-GRASP)))))
    assert results[0] > 20.
    assert results[1] < .01


def test_actual_plant_caps_combined_pd_and_bias():
    p = plant(1.)
    p.arm_kp[:] = 10000.
    p.target_arm = GRASP + .1
    p._applied_arm_target = p.target_arm.copy()
    RsMujocoSync._update_physics(p)
    np.testing.assert_allclose(p._last_arm_tau, [27.,27.,27.,7.,13.,7.])
