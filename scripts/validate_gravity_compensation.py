#!/usr/bin/env python3
"""Hardware-free gravity benchmark using the active arm and production torque law.

Run after sourcing ROS: python3 scripts/validate_gravity_compensation.py
Contacts are disabled ONLY on private test models, to remove environmental support.
Joint limits, finger equality, motor friction, inertia and speed limits remain active.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
for package in ('rebotarm_description', 'rebotarm_planning', 'rebotarm_mujoco_rs'):
    sys.path.insert(0, str(ROOT / 'src' / package))

import mujoco
import numpy as np
import yaml
from rebotarm_description.parameters import ARM_JOINTS as MODEL_JOINTS, arm_gains, joint_limits, load_mounts
from rebotarm_planning.sdk_model import SdkRobotModel
from rebotarm_planning.collision_planner import CollisionPlanner
from rebotarm_mujoco_rs.motor_model import load_motor_model
from rebotarm_mujoco_rs.joint_control import pd_bias_torque, limit_arm_torque, ARM_JOINTS
from rebotarm_mujoco_rs.gravity_scan import _Scanner

SCENE = ROOT / 'src/rebotarm_description/mjcf/grasp_scene.xml'
URDF = ROOT / 'src/rebotarm_description/urdf/rebotarm_with_d435.urdf.xacro'
SPECS = ROOT / 'src/rebotarm_description/config/motor_specs.yaml'
TCP_OFFSET = np.array(load_mounts()["tcp"]["xyz_m"])
FIXED_LIMITS = np.array([joint_limits()[n]["effort"] for n in MODEL_JOINTS])


def addresses(model):
    joints = [model.joint(n) for n in ARM_JOINTS]
    return (np.array([int(j.qposadr[0]) for j in joints]),
            np.array([int(j.dofadr[0]) for j in joints]))


def tcp(model, data):
    body = model.body('gripper_end').id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ TCP_OFFSET


def physics_model(mass_scale=1., com_shift=0.):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # This is a gravity bench, not a grasp/contact simulation.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    motor = load_motor_model(SPECS)
    _, dofs = addresses(model)
    for i, name in enumerate(ARM_JOINTS):
        spec = motor.spec(name)
        model.dof_armature[dofs[i]] += motor.reflected_inertia(name)
        model.dof_frictionloss[dofs[i]] += spec.friction_static
        model.dof_damping[dofs[i]] += spec.friction_viscous
    # All movable robot masses/inertias, excluding fixed base and free scene objects.
    for name in (*[f'link{i}' for i in range(1, 7)], 'gripper_end',
                 'gripper_left', 'gripper_right', 'gripper_coupler', 'd435_link'):
        body = model.body(name).id
        model.body_mass[body] *= mass_scale
        model.body_inertia[body] *= mass_scale
    # Local x shift of wrist and D435 inertial origins; sizes/geometry are unchanged.
    for name in ('gripper_end', 'd435_link'):
        model.body_ipos[model.body(name).id, 0] += com_shift
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model, motor


def run_case(pose_name, q, sdk, out, *, scale=1., gain='low',
             mass_scale=1., com_shift=0., ff='mujoco_bias', seconds=6.,
             target=None, ramp_seconds=0., disturbance_nm=0.):
    model, motor = physics_model(mass_scale, com_shift)
    data = mujoco.MjData(model)
    pos, dofs = addresses(model)
    acts = np.array([model.actuator(n+'_motor').id for n in ARM_JOINTS])
    nominal, _ = physics_model()
    nominal_data = mujoco.MjData(nominal)
    data.qpos[pos] = q
    mujoco.mj_forward(model, data)
    start_tcp = tcp(model, data).copy()
    goal = q.copy() if target is None else target.copy()
    gains = arm_gains()
    kp = np.full(6, 5.) if gain == 'low' else np.array(gains['kp'])
    kd = np.full(6, 2.) if gain == 'low' else np.array(gains['kd'])
    rated = np.array([motor.spec(n).rated for n in ARM_JOINTS])
    peak = np.array([motor.spec(n).peak for n in ARM_JOINTS])
    rows, saturation, limit_hits = [], np.zeros(6), np.zeros(6)
    max_tau, max_req = np.zeros(6), np.zeros(6)
    step_count = round(seconds/model.opt.timestep)
    for step in range(step_count):
        t = step * model.opt.timestep
        fraction = min(t/max(ramp_seconds, model.opt.timestep), 1.)
        # Cubic smoothstep is only used for the optional transfer experiment.
        fraction = fraction*fraction*(3.-2.*fraction)
        command = q + fraction*(goal-q) if ramp_seconds else goal
        measured = data.qpos[pos].copy()
        speed = data.qvel[dofs].copy()
        if ff == 'mujoco_bias':
            nominal_data.qpos[:] = data.qpos
            nominal_data.qvel[:] = data.qvel
            mujoco.mj_forward(nominal, nominal_data)
            bias = nominal_data.qfrc_bias[dofs].copy()
        else:
            bias = sdk.gravity(measured)
            if ff == 'sdk_legacy_155':
                bias[1:3] *= 1.55
        requested = pd_bias_torque(measured, speed, command, kp, kd, bias, scale)
        applied = limit_arm_torque(requested, speed, FIXED_LIMITS, motor)
        data.ctrl[acts] = applied
        master = model.joint('joint7')
        data.ctrl[model.actuator('joint7_motor').id] = np.clip(
            -80.*data.qpos[master.qposadr[0]]-10.*data.qvel[master.dofadr[0]], -150., 150.)
        # Optional external load at J2; testing disturbance recovery, not payload modeling.
        data.qfrc_applied[dofs[1]] = disturbance_nm if 2. <= t < 2.25 else 0.
        saturation += np.abs(requested-applied) > 1e-8
        max_tau = np.maximum(max_tau, np.abs(applied))
        max_req = np.maximum(max_req, np.abs(requested))
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = model.joint(name).range
            limit_hits[i] += measured[i] < lo+.002 or measured[i] > hi-.002
        if step % 10 == 0:
            rows.append([t, *np.rad2deg(measured-command), *applied, *tcp(model, data)])
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or data.time < t:
            raise RuntimeError('nonfinite/reset simulation state')
    mujoco.mj_forward(model, data)
    error = np.rad2deg(data.qpos[pos]-goal)
    name = f'{pose_name}_{gain}_s{scale:g}_{ff}_m{mass_scale:g}_c{com_shift:g}_d{disturbance_nm:g}'
    with (out/f'{name}.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t_sec', *[f'{n}_error_deg' for n in ARM_JOINTS],
                         *[f'{n}_torque_nm' for n in ARM_JOINTS], 'tcp_x_m', 'tcp_y_m', 'tcp_z_m'])
        writer.writerows(rows)
    result = dict(name=name, pose=pose_name, gain=gain, scale=scale, ff=ff,
                  mass_scale=mass_scale, com_shift_m=com_shift, seconds=seconds,
                  target_rad=goal.tolist(), initial_rad=q.tolist(),
                  final_error_deg=error.tolist(), max_final_error_deg=float(np.max(np.abs(error))),
                  final_speed_rad_s=data.qvel[dofs].tolist(),
                  tcp_drift_mm=float(np.linalg.norm(tcp(model,data)-start_tcp)*1000) if target is None else None,
                  max_applied_nm=max_tau.tolist(), max_requested_nm=max_req.tolist(),
                  max_rated_utilization=(max_tau/rated).tolist(),
                  saturation_fraction=(saturation/step_count).tolist(),
                  near_limit_fraction=(limit_hits/step_count).tolist(),
                  kp=kp.tolist(), kd=kd.tolist(), csv=f'{name}.csv')
    print(f"{name}: error={result['max_final_error_deg']:.3f} deg, "
          f"rated={100*max(result['max_rated_utilization']):.1f}%, "
          f"sat={100*max(result['saturation_fraction']):.2f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/validation/gravity')
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=6.)
    args = parser.parse_args()
    if args.samples < 1 or args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error('samples and seconds must be positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    sdk = SdkRobotModel(URDF, 'gripper_tcp')
    scanner = _Scanner(str(SCENE), yaml.safe_load(SPECS.read_text()))
    rng = np.random.default_rng(20261010)
    sampled = rng.uniform(scanner.lo, scanner.hi, (args.samples, 6))
    diffs = np.array([scanner.torque(q, 0.)-sdk.gravity(q) for q in sampled])
    alignment = dict(samples=args.samples, max_abs_nm=np.max(np.abs(diffs), axis=0).tolist(),
                     rms_nm=np.sqrt(np.mean(diffs**2, axis=0)).tolist(), tolerance_nm=.01)
    alignment['passed'] = bool(np.max(np.abs(diffs)) < alignment['tolerance_nm'])
    print('Gravity alignment:', alignment, flush=True)
    # Select collision-free poses away from joint limits, with the hand above the table.
    planner = CollisionPlanner(SCENE)
    candidates = []
    for q in sampled:
        if np.min(np.minimum(q-scanner.lo, scanner.hi-q)) < .12 or not planner.valid(q):
            continue
        point, _ = sdk.fk(q)
        if point[2] > .18:
            candidates.append((float(np.linalg.norm(point[:2])), q))
    if len(candidates) < 2:
        raise RuntimeError('too few collision-free candidate poses')
    candidates.sort(key=lambda item: item[0])
    poses = dict(compact=candidates[0][1], extended=candidates[-1][1],
                 grasp=np.array([-.335855,-1.914513,1.684191,-1.343377,.001031,-1.911958]),
                 lift=np.array([-.335330,-1.813688,1.845187,-1.570796,.001242,-1.911529]))
    motor = load_motor_model(SPECS)
    rated = np.array([motor.spec(n).rated for n in ARM_JOINTS])
    static = []
    for name, q in poses.items():
        for payload in (0., .045, 1.5, 2.5):
            tau = scanner.torque(q, payload)
            static.append(dict(pose=name, q_rad=q.tolist(), payload_kg=payload,
                               torque_nm=tau.tolist(), rated_utilization=(abs(tau)/rated).tolist()))
    worst, worst_q = scanner.scan(.045, args.samples, True, 20261010)
    scan = dict(payload_kg=.045, collision_filtered=False,
                max_abs_nm=worst.tolist(), rated_utilization=(worst/rated).tolist(),
                worst_q_rad=worst_q.tolist(), method='random + differential_evolution (not a proof of global maximum)')
    print('Unfiltered static envelope:', scan, flush=True)
    runs = []
    for name, q in poses.items():
        for scale in (0., .3, 1.):
            runs.append(run_case(name,q,sdk,out,scale=scale,seconds=args.seconds))
    for scale in (0., .3, 1.):
        runs.append(run_case('grasp',poses['grasp'],sdk,out,scale=scale,gain='default',seconds=args.seconds))
    for ff in ('sdk_gravity', 'sdk_legacy_155'):
        runs.append(run_case('grasp',poses['grasp'],sdk,out,ff=ff,seconds=args.seconds))
        runs.append(run_case('grasp',poses['grasp'],sdk,out,ff=ff,gain='default',seconds=args.seconds))
    for mass_scale, com_shift in ((.9,0.),(1.1,0.),(1.,.01),(1.1,.01)):
        runs.append(run_case('grasp',poses['grasp'],sdk,out,mass_scale=mass_scale,
                             com_shift=com_shift,seconds=args.seconds))
        runs.append(run_case('grasp',poses['grasp'],sdk,out,mass_scale=mass_scale,
                             com_shift=com_shift,gain='default',seconds=args.seconds))
    runs.append(run_case('grasp',poses['grasp'],sdk,out,disturbance_nm=1.,seconds=args.seconds))
    # Bench transfer has no contact checking and is not a grasp execution test.
    for scale in (0.,1.):
        runs.append(run_case('grasp_to_lift',poses['grasp'],sdk,out,target=poses['lift'],
                             ramp_seconds=3.,scale=scale,gain='default',seconds=max(args.seconds,6.)))
    files = [SCENE, SCENE.parent/'robot.xml', URDF, URDF.parent/'robot_arm.urdf', SPECS,
             ROOT/'src/rebotarm_mujoco_rs/rebotarm_mujoco_rs/joint_control.py',
             ROOT/'third_party/reBotArm_control_py/reBotArm_control_py/controllers/rebotarm_endpose_controller.py']
    report = dict(seed=20261010, mujoco_version=mujoco.__version__, pinocchio_version=sdk._pin.__version__,
                  contacts_enabled=False, production_bias_scale_default=1.,
                  alignment=alignment, static=static, unfiltered_scan=scan, runs=runs,
                  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    # Ubuntu's matplotlib is built for system NumPy 1.x; physics uses user NumPy 2.x.
    # Plot in the system interpreter, with data exchanged as CSV/JSON.
    env = dict(os.environ, MPLCONFIGDIR='/tmp/rebotarm_gravity_matplotlib')
    env.pop('PYTHONPATH', None)
    subprocess.run(['/usr/bin/python3', '-s', str(ROOT/'scripts/plot_gravity_validation.py'),
                    str(out)], env=env, check=True)
    print(f'Results: {out}/results.json',flush=True)
    if not alignment['passed']:
        raise SystemExit('gravity model alignment failed')


if __name__ == '__main__':
    main()
