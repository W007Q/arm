# reBotArmController_ROS2 项目总览与参数手册

> 本文说明整个项目**怎么运作、怎么用、每个可调参数是什么**。
> 目标场景：**先在仿真里验证运动学/抓取可行，再把同样的 ROS 消息经 UART 发给 STM32 驱动真机**。
>
> 需要逐包、逐文件查找代码入口时，请看 [`src` 目录架构与文件说明](src_architecture_zh.md)。
> 独立球拍模型、一维颠球和 PPO 训练见 [乒乓球仿真与 PPO](pingpong_zh.md)。
> 统一控制接口、MoveIt/Servo 区别及高速颠球路线见 [控制接口说明](control_interfaces_zh.md)。当前默认仍是共享 task_server，可选 ros2_control 路径不自动启动。

---

## 1. 一句话理解

这是一个 **ROS 2 机械臂工作空间**：把底层控制封装成 ROS 话题/服务/动作。
硬件侧有**两条互斥路径**，你按需选一条：

```
                ┌── 路径 A（SDK）: reBotArmController → SDK → motorbridge → CAN/串口 → 电机
上位机(ROS) ────┤
                └── 路径 B（UART）: rebotarm_uart → UART → STM32 → CAN → 电机
     │
     ├── ROS DDS：节点间 topics/services/actions
     ├── HTTP：MCP / LLM（Agent）
     └── 仿真：mujoco_sync（MuJoCo 物理）
```

- **路径 A** 用第三方 SDK `reBotArm_control_py`（pinocchio 做 FK/IK/重力 + motorbridge 通信）。
- **路径 B** 完全不用 SDK：上位机把 MIT 目标经自定义 UART 协议发给 STM32，STM32 负责实时周期/CAN/看门狗。
- **仿真**不需要 SDK 也能跑（MuJoCo FK + scipy IK）；但要让仿真和真机一致，需要**统一模型/求解器**（见 §5）。

---

## 2. 目录与包职责

| 包 | 作用 | 入口/节点 |
|---|---|---|
| `rebotarm_description` | **唯一模型源**：URDF、mesh、MJCF、D435 安装位姿 | — |
| `rebotarm_msgs` | 自定义 msg/srv/action | — |
| `rebotarmcontroller` | 真机驱动（路径 A）+ 假驱动 | `reBotArmController`、`FakeRsDriver` |
| `rebotarm_bringup` | 真机/假驱动启动与硬件配置 | `bringup`、`real_uart`、`fake_rs_bringup` |
| `rebotarm_mujoco_rs` | MuJoCo 物理和模拟相机；共享规划在 planning 包 | `mujoco_rs_sync`、`rs_scene_camera`、`rs_gravity_scan` |
| `rebotarm_vision` | 仿真与 D435 共用 RGB-D 检测 | `rgbd_detector`、`real_d435.launch.py` |
| `rebotarm_tasks` | 共享抓取流程和夹爪等待 | `pick`、`gripper_state` |
| `rebotarm_pingpong` | 独立球拍仿真、一维基线与 PPO | `pingpong` / `python -m rebotarm_pingpong.cli` |
| `rebotarm_uart` | 路径 B：ROS→UART→STM32 | `uart_bridge`、`posvel_to_mit`、`mit_hold` |
| `rebotarm_agent` | MCP/LLM 工具层；抓取 CLI 在 tasks 包 | `rebotarm_mcp_server`、`rebotarm_text_agent` |
| `rebotarm_moveit_config` / `_demos` | MoveIt 2 规划与演示 | `demo.launch.py` 等 |
| `third_party/reBotArm_control_py` | 第三方 SDK（需 clone） | — |
| `third_party/reBot-B601-RS-for-mujoco_sim` | 上游 MuJoCo 模型 | — |

历史独立工程已从名称为单个空格的目录移到 `archive/legacy_b601_mujoco`；迁移前的重复模型保存在 `archive/pre_description_migration`，确认新结构稳定后再决定是否删除。`build/`、`install/`、`log/` 是 colcon 生成物，不是源码。

### 2.1 你自己的 URDF 应替换在哪里

唯一替换入口是：

```text
src/rebotarm_description/urdf/robot_arm.urdf
```

CAD2URDF 导出后，把主 URDF 放到这个路径，并把 mesh 放到：

```text
src/rebotarm_description/meshes/
```

URDF 中 mesh URI 必须写成：

```xml
package://rebotarm_description/meshes/你的网格文件.STL
```

不要替换 `rebotarm_with_d435.urdf.xacro`。新 CAD 已在主 URDF 中定义 `camera_link/camera_joint`；这个包装文件只增加 TCP、可选相机碰撞网格和传感器/光学坐标，避免重复创建相机 link。旧路径 `rebotarm_bringup/description/urdf/00-arm-rs_asm-v3.urdf` 只是兼容符号链接。

MuJoCo不直接运行 URDF。新 URDF 验证完成后，还要把它转换并更新：

```text
src/rebotarm_description/mjcf/robot.xml
```

抓取场景入口为：

```text
src/rebotarm_description/mjcf/grasp_scene.xml
```

转换后必须补回 actuator、关节阻尼、接触几何、夹爪联动和力矩限制。不能只替换 URDF 就认为 MuJoCo 已使用新机械臂。

#### STEP 经 CAD2URDF 转换的建议流程

1. 在 SolidWorks 中把装配体整理成真实运动链。每个刚体只保留一个 link；旋转副轴线必须与实际电机输出轴重合；坐标单位统一为 m、kg、rad。不要把整个装配体合并为一个实体。
2. 在 CAD 中填写真实材料或实测质量，并核对每个部件的质量、质心和惯量。夹爪、D435 支架、线缆和球拍等末端负载必须计入对应 link 或单独建 link。
3. 导出 STEP 前将机械臂放在定义好的零位，固定底座，简化螺钉、倒角等无关细节；另行准备低面数碰撞模型。在线转换涉及上传 CAD 文件，若模型有保密要求，应先确认可接受网站的数据处理方式。
4. 打开 [CAD2URDF 上传页](https://cad2urdf.stackforce.cc/upload)，上传 STEP，按 `base_link → link1 … link6 → gripper_end → gripper_left/right` 建立父子关系，并逐个设置 revolute/prismatic/fixed、轴向、原点和上下限。网站界面若与本文不同，以页面当前必填项为准。
5. 下载结果后先在临时目录解压。把主文件重命名为 `robot_arm.urdf`，把 mesh 复制到 `src/rebotarm_description/meshes/`，将所有 mesh URI 改为 `package://rebotarm_description/meshes/...`，然后替换本文 §2.1 指定的唯一入口。
6. 保留代码依赖的六个手臂关节名 `joint1`…`joint6` 和末端帧 `gripper_end`。夹爪建议保留 `gripper_joint1/2`、`gripper_left/right`；若改名，必须同步 MoveIt、硬件 YAML、MuJoCo 和任务节点中的映射。
7. 用下面命令做结构检查，再在 RViz 检查零位、轴向、比例和碰撞几何。质量不能为 0，惯量矩阵应为正定，关节 effort/velocity/limit 不能沿用转换器占位值。

```bash
source /opt/ros/humble/setup.bash
cd ~/reBotArmController_ROS2-main
colcon build --symlink-install --packages-select rebotarm_description
source install/setup.bash
xacro src/rebotarm_description/urdf/rebotarm_with_d435.urdf.xacro > /tmp/rebotarm.urdf
check_urdf /tmp/rebotarm.urdf
```

8. URDF 通过后运行 `python3 scripts/sync_robot_parameters.py` 和 `--check`，自动更新 MJCF 的关节/惯量参数及 UART 限位；碰撞代理和新结构仍需核验。

### 2.2 D435 末端安装链

当前 TF 结构为：

```text
base_link → cad_base_link → ... → gripper_end → camera_joint → camera_link
                                                └→ gripper_tcp
camera_link → camera_sensor_frame → camera_color_frame → camera_color_optical_frame
                                 └→ camera_depth_frame → camera_depth_optical_frame
```

- `robot_arm.urdf` 保存新 CAD 的相机外壳、固定安装关节、质量和惯量。
- `d435_mount.yaml` 的 `d435_mount` 只描述 `camera_link → camera_sensor_frame`；相机质量不重复填写。`tcp` 描述夹爪中心工具点。
- 仿真由 Xacro 发布内部光学 TF；真机由 RealSense 驱动发布，其根帧为 `camera_sensor_frame`，避免与 CAD `camera_link` 混用。
- 新包的 `camera_joint` 原始值为零，实体位置嵌在 STL 中，因此原点不在镜头上。目前传感器坐标取 CAD 前表面估计值，标记为 `optics_unverified`；视觉抓取前仍需测量镜头坐标并做手眼验证。纯 RViz 路径示教不依赖图像。

新模型适配与离线执行结果见 [验证记录](validation/cad_model_20261026.md)。修改 YAML 后执行 `scripts/sync_robot_parameters.py`，使 Xacro、MuJoCo 和规划使用相同 TCP/传感器坐标。

### 2.3 质量、电机和重力补偿参数

机械结构质量、质心和惯量修改：

```text
src/rebotarm_description/urdf/robot_arm.urdf
```

每个 `<link>` 的 `<inertial>` 中修改 `<mass>`、质心 `<origin>` 和 `<inertia>`。Pinocchio/SDK 真机重力补偿读取这份 URDF；description 中 `rebotarm_hardware.yaml` 的 hybrid 模型已经指向它。

URDF 的关节原点、轴、限位和惯量通过 `scripts/sync_robot_parameters.py` 同步到 MJCF，位置限位同时派生到六轴/七轴 UART 配置。TCP 和相机传感器外参维护在 `description/config/d435_mount.yaml`，相机实体安装与惯量维护在主 URDF。

电机额定/峰值力矩、转速、转子惯量、摩擦、热参数、真机电机映射、MIT 增益、重力补偿方向和 UART 策略统一维护在：

```text
src/rebotarm_description/config/rebotarm_hardware.yaml
```

各关节增益只填写 `models.hybrid.overrides.joints[*].MIT`；电机物理规格在 `motor_catalog`；UART 策略在 `uart`。`motor_specs.yaml`、bringup 硬件 YAML 和 UART YAML 是生成文件，勿手工修改。详细来源表及兼容入口迁移见 [src_architecture_zh.md](src_architecture_zh.md)。

修改参数后：

```bash
python3 scripts/sync_robot_parameters.py
python3 scripts/sync_robot_parameters.py --check
```

重力矩本身主要由 URDF 的质量、质心、惯量和当前关节角决定；电机参数主要决定力矩是否可实现、限幅、摩擦补偿和热安全。建议先令 `tau_scale` 小于 1，从低补偿比例开始逐关节验证方向。

### 2.4 夹爪名称对应关系

| 名称 | 类型与作用 |
|---|---|
| `gripper_end` | 夹爪主体 link，也是 `link6` 后的末端参考坐标系；D435 固定在它下面。它不是运动关节。 |
| `gripper_left` | 左夹指 link，包含左指的外观、碰撞、质量和惯量。 |
| `gripper_right` | 右夹指 link，包含右指的外观、碰撞、质量和惯量。 |
| `joint_left` | MuJoCo 中左夹指的直线关节名；当前 URDF 中对应 `gripper_joint1`。 |
| `joint_right` | MuJoCo 中右夹指的直线关节名；当前 URDF 中对应 `gripper_joint2`。 |

MuJoCo 还定义了主驱动 `joint7`，通过 equality 约束让 `joint_left`、`joint_right` 同步运动。ROS/URDF 对外发布的是 `gripper_joint1`、`gripper_joint2`，任务节点负责这两组名称的映射。

---

## 3. 快速上手

### 3.1 构建
```bash
python3 scripts/sync_robot_parameters.py
colcon build --symlink-install
source install/setup.bash
```

### 3.2 抓取仿真（MuJoCo）

`grasp.launch.py` 是抓取的统一入口；它会同时启动 MuJoCo、视觉模拟、规划器和夹爪状态节点。

```bash
# 终端 1
source /opt/ros/humble/setup.bash
source install/setup.bash
MUJOCO_GL=egl ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=mujoco use_viewer:=false

# 终端 2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run rebotarm_tasks pick --config "$PWD/src/rebotarm_tasks/config/pick_mujoco.yaml" \
  --color red --action-timeout-padding-sec 40
```

需要交互窗口时把 `use_viewer:=false` 改为 `true`。电脑较慢时保持 `false`，避免渲染线程拖慢物理步进。

### 3.3 UART 真机：只通信/保持当前位置

这个入口只启动 UART bridge 和位置适配器，不启动规划器、相机或抓取任务。适合先检查 STM32 的反馈和保持目标。

```bash
# 终端 1：替换为实际的稳定设备路径
./scripts/start_uart_bridge.sh /dev/serial/by-id/<STM32设备>

# 终端 2：发送使能首帧，服务立即返回；随后异步确认运行状态
./scripts/start_uart_bridge.sh enable

# 停止发送并让 STM32 看门狗失能
./scripts/start_uart_bridge.sh disable
```

查看状态：

```bash
ros2 topic echo /rebotarm/uart/status --once
ros2 topic echo /rebotarm/arm_status --once
```

这里不能直接执行抓取；抓取请使用下一节的统一入口。不要同时启动两套 UART bridge。

### 3.4 UART 真机抓取（UART→STM32→CAN）

`backend:=uart` 已经包含 UART bridge、位置适配器、机器人状态发布、规划器和可选 D435。不要再启动 `start_uart_bridge.sh <串口>`。

```bash
# 终端 1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart \
  port:=/dev/serial/by-id/<STM32设备> \
  task_config:="$PWD/src/rebotarm_tasks/config/pick_uart.yaml" \
  planning_config:="$PWD/src/rebotarm_planning/config/planning_hardware.yaml" \
  start_camera:=true gravity_feedforward:=false

# 终端 2：反馈正常、七轴失能且目标已对齐后
./scripts/start_uart_bridge.sh enable
ros2 run rebotarm_tasks pick \
  --config "$PWD/src/rebotarm_tasks/config/pick_uart.yaml" \
  --color red --action-timeout-padding-sec 40
```

停止顺序：先停止 `pick`，再执行 `./scripts/start_uart_bridge.sh disable`，最后退出 launch。重力前馈必须完成实体标定后才改为 `gravity_feedforward:=true`。

标定完成后，重启 UART 抓取入口即可开启重力前馈：

```bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart port:=/dev/serial/by-id/<STM32设备> \
  start_camera:=true gravity_feedforward:=true
```

它仍需单独调用 `./scripts/start_uart_bridge.sh enable`，且只在支撑机械臂、低速验证通过后使用。

### 3.5 SDK/CAN 真机抓取（路径 A）

```bash
# 终端 1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=sdk channel:=can0 \
  task_config:="$PWD/src/rebotarm_tasks/config/pick_sdk.yaml" \
  planning_config:="$PWD/src/rebotarm_planning/config/planning_hardware.yaml" \
  start_camera:=true

# 终端 2：确认反馈正常后
ros2 service call /rebotarm/enable std_srvs/srv/Trigger '{}'
ros2 run rebotarm_tasks pick \
  --config "$PWD/src/rebotarm_tasks/config/pick_sdk.yaml" \
  --color red --action-timeout-padding-sec 40
```

SDK/CAN 和 UART 只能选一个，不能同时启动。

### 3.6 RViz/MoveIt 示教

仿真：

```bash
ros2 launch rebotarm_moveit_config moveit_sim.launch.py
```

真机：先启动 UART 和轨迹服务，纯示教关闭相机：

```bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart port:=/dev/ttyCH343USB0 \
  start_camera:=false gravity_feedforward:=false
```

再开第二个终端：

```bash
ros2 launch rebotarm_moveit_config hardware.launch.py \
  arm_namespace:=rebotarm use_rviz:=true
```

调用 `./scripts/start_uart_bridge.sh enable` 发出首帧后，等待 UART 状态中的 `enabled=true`（表示后台已确认七轴运行），再在 RViz 的 `MotionPlanning` 面板中拖动末端标记，点击 **Plan** 预览，再点击 **Execute** 执行。纯示教不要求相机或手眼标定；新 CAD 的 D435 外壳碰撞在 MoveIt 和执行端默认开启，暂停图像流不会关闭实体碰撞。真实执行仍要求 URDF 零位/方向与实物一致，并确认桌面及固定障碍物的位置。`hardware_scene_calibrated=false` 会阻止轨迹执行，见 [抓取执行与真机迁移](grasp_execution_zh.md)。

### 3.7 末端 D435 单独启动

只有在没有使用 `grasp.launch.py start_camera:=true` 时才单独启动相机：

```bash
ros2 launch rebotarm_vision real_d435.launch.py \
  arm_namespace:=rebotarm base_frame:=base_link
```

该入口需要已有 `robot_state_publisher` 和 `/rebotarm/joint_states`，否则相机坐标无法变换到 `base_link`。

### 3.8 乒乓球仿真与 PPO

```bash
export PYTHONPATH="$PWD/src/rebotarm_description:$PWD/src/rebotarm_pingpong:$PYTHONPATH"
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli baseline --episodes 3
python3 -m rebotarm_pingpong.cli view
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli train --steps 200000 --envs 2 --output artifacts/pingpong/my_run
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli evaluate --model artifacts/pingpong/my_run/final_model.zip --output artifacts/pingpong/my_eval
```

颠球环境独立于抓取后端，不会向 UART/CAN 真机发命令。

### 3.9 电机选型（静态重力矩扫描）

```bash
ros2 run rebotarm_mujoco_rs rs_gravity_scan --output /tmp/scan.csv
```

旧的 `fake_rs_bringup`、`real_uart.launch.py` 和 `grasp_sim.launch.py` 仍保留作兼容/底层调试；新项目入口统一使用 `grasp.launch.py backend:=mujoco|uart|sdk`。

---

## 4. 数据流

### 4.1 仿真抓取
```
D435 RGB-D → rgbd_detector → pick_color → task_server(IK/轨迹) → arm/command/position → 被控对象
被控对象 → joint_states → task_server / agent / RViz
```

### 4.2 真机（路径 B）
```
task_server → arm/command/position → posvel_to_mit → /uart/mit_targets → uart_bridge
           → UART(帧) → STM32 → CAN → 电机
STM32 → UART(反馈帧) → uart_bridge → joint_states / arm_status / uart/status
```

---

## 5. 统一 FK/IK/重力（仿真 ↔ 真机）

真机（路径 A/STM32 上游）用 SDK 的 pinocchio；仿真默认用 MuJoCo + scipy。要**让仿真结果能直接用于真机**，把仿真的 `rs_task_server` 切到 SDK 后端：

```bash
ros2 launch rebotarm_mujoco_rs mujoco_rs.launch.py kinematics_backend:=sdk
```

- `kinematics_backend=mujoco`（默认）：MuJoCo FK + scipy IK；
- `kinematics_backend=sdk`：SDK(pinocchio) FK + CLIK IK，与真机一致；
- 已实测：同模型下 **FK 完全一致**；**IK 解在 1e-6 内一致**；重力差约 0.1 Nm（MuJoCo 手转模型的质量略有差异）。

> 前提：`third_party/reBotArm_control_py` 已 clone，且装了 `pinocchio` 与 `motorbridge`。

---

## 6. 参数总览

### 6.1 `rebotarm_bringup`（真机路径 A / 假驱动）

**`bringup.launch.py` / `driver.launch.py`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | `""` | 旧入口兼容参数；新入口固定使用 `hybrid` |
| `channel` | `""` | `/dev/ttyACM0` 或 `can0` |
| `hardware_config` | bringup 的 YAML | 硬件配置路径 |
| `joint_state_rate` | `100.0` | `/joint_states` 频率 (Hz) |
| `cmd_arbitration` | `reject` | 轨迹中低层命令策略：`reject`/`preempt` |
| `arm_namespace` | `rebotarm` | 命名空间 |
| `frame_id` / `ee_frame_id` | `base_link`/`end_link` | TF 帧名（预留） |
| `disable_after_safe_home` | `true` | 回安全位后是否失能 |
| `use_rviz` | `false` | 仅 bringup |

**`fake_rs_bringup.launch.py`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `arm_namespace` | `rebotarm` | 命名空间 |
| `joint_state_rate` | `100.0` | 状态频率 |
| `max_joint_speed` | `1.0` | 假驱动关节最大速度 (rad/s) |
| `max_gripper_speed` | `2.0` | 夹爪最大速度 |
| `start_enabled` | `true` | 启动即使能 |
| `use_rviz` | `false` | RViz |

### 6.2 `rebotarm_hardware.yaml`（硬件/模型/增益）

| 参数 | 说明 |
|---|---|
| `default_model` | 当前为 `hybrid`（DM/PA43 混合机械臂） |
| `models.<m>.sdk_config` | SDK 侧基础配置文件名 |
| `overrides.urdf_path` | FK/IK/重力用的模型路径 |
| `overrides.end_effector_frame` | 末端帧（DM `end_link` / RS `gripper_end`） |
| `overrides.channel` | 通信口 |
| `overrides.rate` | 控制环频率 |
| `overrides.groups.arm.joints` / `gripper.joints` | 关节分组 |
| `overrides.gripper.position_limits.open/close` | 夹爪开/合位置（DM open=-5.0，RS open=5.0） |
| `overrides.control.arm_control_mode` | `posvel`(DM) / `mit`(RS) |
| `overrides.joints[*].MIT.kp/kd` | 每个逻辑关节唯一的 MIT 位置刚度/阻尼来源；仿真、SDK、UART 共用 |
| `overrides.gravity_compensation.kp` / `kd` | 重力补偿时的弱位置环增益 |
| `overrides.gravity_compensation.joint_direction` | 每关节 ±1（方向不对会反向甩飞） |
| `overrides.gravity_compensation.tau_scale` | 每关节重力矩缩放（1.0 全补偿） |

### 6.3 `rebotarm_mujoco_rs`（仿真）

| 参数 | 默认 | 说明 |
|---|---|---|
| `arm_namespace` | `rebotarm` | 命名空间 |
| `model_path` | `""` | MuJoCo 模型路径 |
| `simulation_mode` | `physics` | `kinematic`/`physics` |
| `update_rate` | `250.0` | 仿真更新率 |
| `smoothing_alpha` | `1.0` | kinematic 平滑 |
| `stale_timeout` | `1.0` | 输入超时 |
| `use_viewer` | `false` | MuJoCo 窗口 |
| `arm_kp` / `arm_kd` | 由 `overrides.joints[*].MIT` 自动生成 | 臂阻抗刚度/阻尼；不要在生成的 YAML 中手改 |
| `arm_tau_limit` | `[36,36,36,14,14,14]` | 臂力矩上限 |
| `motor_specs` | 内置 | 电机规格 YAML |
| `motor_model_enabled` | `true` | 启用电机模型（扭矩-转速/热） |
| `compliance_enabled` | `false` | 柔顺模式 |
| `compliance_scale` | `0.5` | 柔顺时刚度缩放 |
| `contact_torque_threshold` | `1.0` | 接触判定阈值 (Nm) |
| `authority_mode` | `false` | 仿真即真机（发 `/joint_states`） |
| `joint_states_topic` | `""` | 权威模式发布话题 |
| `subscribe_commands` | `false` | 仿真直接收六轴 `ArmPositionCommand`；authority 模式自动开启 |
| `legacy_joint_command_topics` | `false` | 显式切回旧逐轴话题，与批量接口互斥 |
| `gripper_kp` / `gripper_kd` | `80` / `10` | 夹爪 PD；抓取场景覆盖为 `120` / `8` |
| `gripper_tau_limit` | `150.0` | 夹爪力矩上限 |
| `gripper_force_limit` | `0.0` | 夹爪力限幅（0=不限） |
| `provide_gripper_services` | `false` | 仿真提供夹爪服务 |
| `control_mode` | `joint` | `joint`/`cartesian` |
| `cartesian_target_topic` | `""` | 笛卡尔目标话题 |
| `cartesian_kp/kd/rot_kp/rot_kd` | `200/20/20/2` | 笛卡尔阻抗增益 |
| `kinematics_backend` | `mujoco` | `mujoco`/`sdk`（统一 FK/IK） |
| `sdk_urdf` / `sdk_end_effector_frame` | `""` / `gripper_end` | SDK 后端模型 |
| `enable_task_tools` | `true` | 是否起 task_server/相机/检测 |

### 6.4 `motor_specs.yaml`（电机模型与选型）

此文件由 description 的硬件配置自动生成。修改额定/峰值力矩和物理规格时进入 `rebotarm_description/config/rebotarm_hardware.yaml` 的 `motor_catalog`，再运行同步脚本。

| 段 | 字段 | 说明 |
|---|---|---|
| `joints.<j>` | `motor`、`rated`、`peak` | 关节对应电机与额定/峰值力矩 (Nm) |
| `motors.<m>` | `mass`、`gear`、`torque_constant`、`no_load_speed`、`rated_speed`、`rotor_inertia`、`friction_static`、`friction_viscous`、`phase_resistance`、`thermal_resistance`、`thermal_capacity` | 电机物理/热参数 |
| `limits` | `jointN: [lo, hi]` | 扫描限位覆盖 |
| `payload` | `body`、`masses_kg` | 负载体与质量档 |
| `scan` | `samples`、`refine`、`seed` | 扫描设置 |
| `ambient_temperature` | 环境温度 | 热模型 |

### 6.5 `rebotarm_uart`（路径 B）

**`uart_bridge`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `port` | `""` | **必填**，STM32 串口路径 |
| `baudrate` | `115200` | 固定 72/131 字节协议的波特率 |
| `arm_namespace` | `rebotarm` | 命名空间 |
| `command_source` | `batch` | `batch`（整组）或 `per_joint` |
| `tx_hz` | `50.0` | 七轴命令发送频率 |
| `command_timeout` | `0.30` | 上游目标超时 → 停止发包；必须小于 STM32 看门狗时间 |
| `feedback_timeout` | `0.25` | 反馈超时 |
| `start_tolerance` | `0.15` | 使能前目标与实测的最大偏差 (rad) |
| `feedback_motor_ids` | `[1,2,8,9,20,1,2]`（十六进制 `[0x01,0x02,0x08,0x09,0x14,0x01,0x02]`） | 当前固件反馈块中的本地 Motor ID；逻辑槽位仍是 1..7 |
| `slot_joint_ids` | `[4,6,1,2,3,5,7]` | STM32 线槽位到逻辑关节号的映射 |
| `slot_feedback_ids` | `[4,5,17,18,19,6,7]` | STM32 CAN 反馈/Master ID；仅作配置和诊断元数据 |
| `joint_names` / `joint_ids` | joint1..joint6, gripper / 1..7 | ROS 逻辑关节顺序 |
| `position_min/max` | URDF/夹爪配置派生 | 实体零点与方向标定后才对应真实机械限位 |
| `velocity_limit` / `torque_limit` | 逻辑关节顺序 `[0.3,0.3,0.3,0.3,0.5,0.3,0.5]` / `[1,1,1,7,3,7,3]` | 已烧录固件和电机峰值共同得到的逐关节限幅 |
| `kp_limit` / `kd_limit` | 固件逐轴上限 / 5 | UART 线格式和已烧录固件的增益上限 |

UART 只使用七轴固定协议，Kp≤65.535、Kd≤5；共享 SDK 中的 Kp=150 不能直接用于此格式。当前 STM32 命令看门狗为 500 ms，72 字节帧不携带超时参数。

**`posvel_to_mit`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `arm_namespace` | `rebotarm` | 命名空间 |
| `joint_names` / `joint_ids` | joint1..joint6, gripper / 1..7 | 与 uart_bridge 的逻辑关节顺序一致 |
| `arm_kp` / `arm_kd` | `joints[].MIT` 按逻辑关节生成 | 转成 MIT 时使用的 UART 刚度/阻尼 |
| `arm_velocity_limit` | 固件逐轴限速 | 传给 STM32 的位置轨迹限速；填 0 会暂停该轴轨迹 |
| `gravity_feedforward` | `false` | 是否用 SDK 加 `tau=g(q)` 前馈 |
| `sdk_urdf` | 统一的 `robot_arm.urdf` | Pinocchio 重力前馈模型；标准 launch 已显式传入 |
| `target_timeout` | `0.35` | 目标新鲜度；必须小于 STM32 看门狗时间 |

### 6.6 `rebotarm_agent` / 一键抓取

| 参数 | 默认 | 说明 |
|---|---|---|
| `arm_namespace` | `rebotarm` | 命名空间 |
| `motion_mode` | `locked` | `locked`/`allow`（运动工具开关） |
| `host` / `port` | `127.0.0.1` / `8081` | MCP 地址 |
| `python_executable` | 外部 venv | 需覆盖为 `/usr/bin/python3` 等 |

`ros2 run rebotarm_tasks pick`：`--namespace`、`--color`(auto/red/blue/yellow)、`--approach-z`、`--grasp-z`、`--lift-z`、`--duration`、`--service-timeout-sec`、`--action-timeout-padding-sec`、`--ready-timeout-sec`。

---

## 7. 关键机制

- **状态机**（真机）：`IDLE / TRAJ_RUNNING / LOWLEVEL_STREAMING / GRAVITY_COMP / SAFE_HOMING`；互斥（重力补偿与轨迹/夹爪不能同时）。
- **回调组**：高频只读（joint_states/动作/透传）用 reentrant 并发；改状态（enable/safe_home/重力）用 mutually-exclusive 串行；`_cmd_lock` 在硬件层兜底。
- **UART 发包与保护**：使能首帧写入成功即返回，七轴新鲜运行反馈由后台异步确认；确认前仅发零输出帧。正常运动仍要求新鲜目标、反馈与限位。确认超时、上游超时或反馈丢失时停发并锁定故障，由 STM32 本地看门狗完成失能。
- **MIT 控制律**：`T_out = Kp·(θref−θ) + Kd·(Vref−V) + Tref`；纯位置 `Tref=0`，重力补偿 `Tref=g(q)`。

---

## 8. 常见问题

| 现象 | 原因/处理 |
|---|---|
| `FileNotFoundError: reBotArm_control_py` | 未 clone SDK；见 §3 或 README |
| `ModuleNotFoundError: pinocchio / motorbridge` | `sudo apt install ros-humble-pinocchio` + `pip install --user motorbridge` |
| 真机驱动起不来 | 路径 A 需要 SDK+motorbridge；否则用路径 B（`rebotarm_uart`） |
| `goal in collision` | 目标/路径与场景物体碰撞（task_server 碰撞规划器） |
| 抓取失败 `verify_physical_lift` | D435 未确认物体抬升至少 35 mm；检查 TCP、抓取高度、接触面和夹持力 |
| 抓取失败 `verify_sustained_grasp` | 物体短暂抬起后滑落；降低抓取点、保持垂直抬升并检查双指接触中心 |
| `Ignoring unexpected goal response` | 同一 ROS domain 存在多套 `rs_task_server`；关闭遗留的 `grasp_sim.launch.py` 实例，只保留一套 action server。`start_camera_sim.sh` 已按 `ROS_DOMAIN_ID` 加单实例锁 |
| 重力补偿反向甩飞 | `gravity_compensation.joint_direction` 配错 |
| 仿真与真机不一致 | 统一 URDF + `kinematics_backend:=sdk` |

---

## 9. 文档索引

- `docs/uart_protocol_zh.md`：UART 线协议（帧/消息/带宽）
- `docs/uart_host_guide_zh.md`：UART 上位机操作
- `docs/simulation_upgrade_zh.md`：仿真升级说明
- `README.md` / `README_zh.md`：安装与 MoveIt 使用
- `API_zh.md`：ROS API 参考
