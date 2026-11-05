# rebotarm_mujoco_rs

MuJoCo device adapter for the workspace's single DM4340P / DM4310 / PA43 arm
model in `rebotarm_description`. Legacy `rs_*` executable names remain for
compatibility.

The package owns physics, measured state, simulated D435 rendering, contact
evidence and motor diagnostics. Shared IK, collision planning and trajectory
execution live in `rebotarm_planning`; colour pick tasks live in
`rebotarm_tasks`. Simulation does not open CAN or UART.

Model/actuator parameters are maintained in `rebotarm_description`: the URDF
owns joint limits and body inertias, `config/rebotarm_hardware.yaml` owns the
motor map/catalog and default gains, and `config/d435_mount.yaml` owns TCP/sensor-frame
mounts. After changing a source, run `python3 scripts/sync_robot_parameters.py`
from the workspace root. `--check` detects stale derived values. The local
`config/motor_specs.yaml` is a generated compatibility copy; the default reader
uses the description package. Collision proxies and contacts stay in MJCF.

Legacy planning/vision/task forwarding modules are for existing external
clients. New imports use `rebotarm_planning`, `rebotarm_vision` and
`rebotarm_tasks`; migration mappings are in `docs/src_architecture_zh.md` §7.1.

Independent paddle simulation and residual PPO live in `rebotarm_pingpong`.
See [pingpong guide](../../docs/pingpong_zh.md). Its scenes are not loaded into
the grasp synchronizer; the old `rs_pingpong_controller` is a sinusoidal demo.

Start the unified workflow with:

```bash
ros2 launch rebotarm_bringup grasp.launch.py backend:=mujoco
ros2 run rebotarm_tasks pick --color red --action-timeout-padding-sec 40
```

Modes:

- `kinematic`: directly synchronizes ROS joint state into MuJoCo.
- `physics` (default): tracks ROS targets with rate-limited position targets,
  conservative PD plus MuJoCo bias forces, motor torque limits, and contact
  diagnostics. Use `kinematic` only for fast interface/MoveIt smoke tests.

`gravity_bias_scale` defaults to `1.0` in both joint and Cartesian physics control.
Set it to `0.0` to disable bias feedforward or `0.3` for partial compensation.
It scales MuJoCo `qfrc_bias`, including velocity bias terms as well as gravity.
The unified MuJoCo launch accepts the same argument; hardware settings are separate.

For a hardware-free comparison, source ROS and run from the workspace root:

```bash
python3 scripts/validate_gravity_compensation.py
```

The benchmark uses the production PD/bias composition and motor limits, disables
environmental contacts only on private models, and saves CSV/JSON/plots under
`docs/validation/gravity/`. See `docs/gravity_compensation_validation_zh.md` for
results and the limits of static, short-duration simulation evidence.

The physics bridge deliberately keeps the ROS-facing target semantics close to
the real low-level target path: arm targets are rate-limited by `arm_target_speed`, the
gripper slide target is rate-limited by `gripper_target_speed` (metres per
second; 2 rad/s at the real 0..5 rad motor range maps to 0.02 m/s), and a stale command
stream makes the simulated robot hold its measured position while continuing
to publish state and diagnostics.

The default arm gains follow the mixed DM/PA43 configuration. They are model
defaults, not a substitute for measured motor and load calibration.

```bash
ros2 launch rebotarm_mujoco_rs mujoco_rs.launch.py \
  arm_namespace:=rebotarm simulation_mode:=physics use_viewer:=true
```

For a standalone simulated controller (without `FakeRsDriver`), enable
authority mode. It publishes `/joint_states` and consumes the same six-axis
`ArmPositionCommand` topic as the UART/SDK adapters:

```bash
ros2 launch rebotarm_mujoco_rs mujoco_rs.launch.py \
  arm_namespace:=rebotarm authority_mode:=true simulation_mode:=physics
```

Important topics:

- `/rebotarm/arm/command/position` — absolute `joint1..joint6` positions and
  non-negative trajectory velocity limits; zero pauses that axis's reference.
  `legacy_joint_command_topics:=true` explicitly selects the old per-joint
  topics instead. The two sources are mutually exclusive.
- `/rebotarm/mujoco/object_states`
- `/rebotarm/mujoco/d435/image_raw`
- `/rebotarm/mujoco/d435/depth`
- `/rebotarm/mujoco/d435/camera_info`
- `/rebotarm/mujoco/d435/pose`
- `/rebotarm/vision/color_blocks/detections`

Task endpoints:

- `/rebotarm/move_to_pose_ik`
- `/rebotarm/move_to_pose`
- `/rebotarm/follow_joint_trajectory`

## Camera grasping and collision-aware execution

The detector uses synchronized RGB-D images and the pose of the D435 attached
to `gripper_end`; it does not consume `object_states`. The nominal simulation
mount must be replaced by a measured bracket transform and eye-in-hand
calibration before hardware use.
The task server uses a known MuJoCo collision map, sampled edge checking and
bounded RRT search, with measured-state completion checks. See the workspace
`docs/simulation_upgrade_zh.md` for scope, startup and limitations, and
`docs/uart_protocol_zh.md` for the proposed STM32 UART protocol.
