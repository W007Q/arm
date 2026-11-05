# 抓取执行与真机迁移

## 1. 启动入口

### 1.1 抓取仿真

仿真终端：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export MUJOCO_GL=egl
ros2 launch rebotarm_bringup grasp.launch.py backend:=mujoco use_viewer:=false
```

另一个终端（同一 ROS_DOMAIN_ID）执行共享任务：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run rebotarm_tasks pick --color red --action-timeout-padding-sec 40
```

本机的 MuJoCo/Pinocchio 等 pip 依赖安装在用户 site-packages。如果当前解释器找不到它们，可添加 `/home/wq/.local/lib/python3.10/site-packages` 到 `PYTHONPATH`，或使用 `scripts/start_camera_sim.sh` 自动加载。

### 1.2 UART 真机抓取

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart \
  port:=/dev/serial/by-id/实际设备 \
  start_camera:=true \
  gravity_feedforward:=false
```

当前 STM32 的 UART 线槽位按电机组排列；上位机已经按
`[J4,J6,J1,J2,J3,J5,J7]` 映射到实体关节，详见
[`docs/uart_protocol_zh.md`](uart_protocol_zh.md)。这只解决通信顺序，不代替零位、方向、限位和力矩标定。

另一个终端直接执行 enable；首次发送不读取启动前的 CAN 状态，只发送零输出握手帧：

```bash
./scripts/start_uart_bridge.sh enable
ros2 run rebotarm_tasks pick \
  --config "$PWD/src/rebotarm_tasks/config/pick_uart.yaml" \
  --color red --action-timeout-padding-sec 40
```

#### 1.2.1 YOLO + D435 目标检测

抓取入口现在默认使用 YOLO；RGB/HSV 颜色分割仍保留为备用后端，只有显式设置
`detector_backend:=color` 才会启用。YOLO 不替代 D435 深度、时间戳或 `base_link`
TF：YOLO 只给出类别和二维框，三维位置仍由同帧深度反投影得到。

YOLO 代码位于 [`yolo_detection.py`](../src/rebotarm_vision/rebotarm_vision/yolo_detection.py)，
ROS 节点位于 [`rgbd_detector.py`](../src/rebotarm_vision/rebotarm_vision/rgbd_detector.py)。
颜色备用代码仍在 [`rgbd_detection.py`](../src/rebotarm_vision/rebotarm_vision/rgbd_detection.py)。

默认读取自己训练的 **`models/best.pt`**，使用 **GPU 0**。从工作空间根目录启动，
训练后将 `runs/detect/<训练名称>/weights/best.pt` 复制到 `models/best.pt`，或通过
`yolo_model:=/绝对路径/best.pt` 指定原文件。节点不会用官方通用模型代替缺失的权重。

当前环境为 Python 3.10、`torch 2.7.1+cu128`、`torchvision 0.22.1+cu128`，
以及 `ultralytics 8.4.189`。YOLO26 通过 `ultralytics` 提供，不是单独的
`yolo26` pip 包。在其他机器上，先于运行 ROS 的同一个 Python 环境安装支持本机
GPU 的 torch/torchvision 配对，再安装视觉依赖：

```bash
python3 -m pip install --user -r src/rebotarm_vision/requirements-yolo.txt
```

RTX 5060 需要支持 `sm_120` 的 PyTorch CUDA 版本，旧的 2.4.0+cu121 不适用。
当前默认 `yolo_device:=0`；启动时会预热模型并验证实际推理，CUDA 错误会明确报出。
仅在显式设置 `yolo_device:=cpu` 时才使用 CPU。不要再安装旧的 `torchvision==0.19.0`。
常用调参是 `yolo_confidence:=0.45`、`yolo_imgsz:=640`：小目标可适当增大推理尺寸，
但会增加显存占用和推理耗时。预热与相机推理使用同一个尺寸。

本机已用官方 YOLO26n 在 `cuda:0` 验证推理，并通过合成深度/位姿验证 ROS 检测输出。
自定义 `best.pt` 和真实 D435 精度需训练后验证。本机 16GB 内存、无交换空间，
整套 MuJoCo + YOLO 启动验证遇到系统内存不足，尚未完成；联合运行前应留足内存。

启动时选择 YOLO，并把类别映射到当前抓取任务的尺寸/颜色配置。下面示例把模型类别
`red_cube` 映射到现有 `red` 抓取配置：

```bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=mujoco use_viewer:=false \
  detector_backend:=yolo \
  yolo_model:="$PWD/models/best.pt" yolo_device:=0 \
  yolo_class_map:='{"red_cube":"red","blue_cup":"blue"}'
```

真机只替换后端和串口：

```bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart port:=/dev/serial/by-id/实际设备 \
  start_camera:=true gravity_feedforward:=false \
  detector_backend:=yolo detection_source:=yolo_rgbd \
  yolo_device:=0 \
  yolo_model:="$PWD/models/best.pt" \
  yolo_class_map:='{"red_cube":"red","blue_cup":"blue"}'
```

先检查检测结果，不要立即执行抓取：

```bash
ros2 topic echo /rebotarm/vision/color_blocks/detections --once
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

未提供类别映射时，目标保留 `best.pt` 中的类别名，可用 `pick --color auto` 选择目标。
继续使用 `pick --color red` 时，需要让模型类别名为 `red`，或配置上面的类别映射。
现有 `/rebotarm/vision/color_blocks/detections` 话题名仍兼容任务节点，内容来自 YOLO；
话题名中的 `color_blocks` 不表示正在使用颜色分割。

检测节点会对每个 YOLO 框的内部区域取 D435 深度中值，反投影到相机坐标，再按同一
帧时间戳的 TF 转到 `base_link`。深度无效或 TF 不连通时会丢弃该目标，不会用真值或旧
坐标补齐。普通检测框适合规则物体；遮挡明显或外形不规则时应使用 YOLO-seg，并在
后续版本把分割掩膜用于深度采样。

停止时先停止任务，再执行 `./scripts/start_uart_bridge.sh disable`。不要同时再启动 `start_uart_bridge.sh <串口>`，因为 `grasp.launch.py` 已经包含 UART bridge。

### 1.3 SDK/CAN 真机抓取

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=sdk channel:=can0 start_camera:=true
```

然后调用 `/rebotarm/enable`，再使用 `pick_sdk.yaml`：

```bash
ros2 service call /rebotarm/enable std_srvs/srv/Trigger '{}'
ros2 run rebotarm_tasks pick \
  --config "$PWD/src/rebotarm_tasks/config/pick_sdk.yaml" \
  --color red --action-timeout-padding-sec 40
```

任务配置用 `task_config:=/绝对路径/pick_uart.yaml`，规划配置用 `planning_config:=/绝对路径/planning_hardware.yaml` 覆盖。CLI 必须使用与 backend 相同的 `pick_uart.yaml` 或 `pick_sdk.yaml`，不能使用默认仿真配置。

完整抓取要求真机配置 `hardware_calibrated=true`、有效 PA43 接触阈值和限矩，以及规划配置 `hardware_scene_calibrated=true`。这些开关是完成实体测量后的声明，不代替标定。默认值为 false，力矩阈值为零，防止把仿真 2 N 误当成 PA43 2 Nm。

只启动一个 backend。新 SDK 入口关闭驱动自带的机械臂 IK/Action，由 `rebotarm_planning` 独占 `move_to_pose_ik`、`move_to_pose` 和 `follow_joint_trajectory`。旧 bringup 与 MoveIt 示例仍是独立工作流，不应和新入口重复提供同名服务。

### 1.4 启动失败的先后检查

如果 `enable` 返回 `STM32 is not in disabled state`，先确认没有另一套 bridge 在发包，执行一次：

```bash
./scripts/start_uart_bridge.sh disable
sleep 1
ros2 topic echo /rebotarm/uart/status --once
```

确认 JSON 中 `feedback_ready=true`、各组已初始化、三组运行位为 0，且 `tx_frames` 停止增加后，再调用 `enable`。新鲜反馈中仍有使能轴时会拒绝启动；过期缓存中的旧使能位不作为当前运行证据。

诊断中的 `feedback_ready` 表示 UART 整包新鲜，`feedback_fresh` 表示各轴 CAN 测量新鲜。enable 首帧不要求这两项；握手发送后必须收到新鲜反馈，才会返回成功并进入正常运行。
断电后 JSON 仍保留最后一帧；`feedback_ready=false` 时，`motor_status` 和 `feedback_state` 只是历史数据。
首次发送采用零限速、零 Kp/Kd、零前馈帧；等待本次发送后的七轴新鲜运行反馈。默认 2 秒内确认失败会停发；确认成功后才接入正常目标，并用新测量检查起始位置偏差。

提供的 `te (2).zip` 在 `IDLE` 且无新命令时不主动查询电机，因此失能时 `feedback_fresh=false` 不一定表示接收故障。若发送握手帧后仍没有新鲜 CAN 反馈，再检查反馈/Master ID、CAN 帧格式和接收中断；不要将缓存强制标记为新测量。

## 2. 真机抓取标定

真机抓取配置默认关闭，是因为下列数据必须来自实体测量。不能只把两个布尔值改成 `true`。

纯 RViz 示教请用 `grasp.launch.py backend:=uart ... start_camera:=false` 启动后端与轨迹服务，无需 D435 或手眼标定；此模式也会关闭相机碰撞代理。拖动目标、Plan 预览不要求真实场景标定；Execute 会调用同一个轨迹服务，仍检查 `hardware_scene_calibrated`。桌面/固定障碍物的位置可直接测量，标定是把这些尺寸换算到 `base_link` 坐标系，不一定要用相机。电机上电置零也不等于已确认 URDF 的零姿态和关节方向。

若启用真实相机，TF 应为 `base_link → 各关节 → gripper_end → camera_link → camera_color_optical_frame`。使能前没有新鲜关节反馈时，机械臂的动态 TF 尚未建立，RGB-D 会报告坐标树不连通；反馈过期后也不能继续使用旧变换定位。后端负责机械臂和安装 TF，RealSense 负责内部光学 TF，MoveIt 硬件入口不重复发布。

### 2.1 标定文件

| 数据 | 修改文件 | 完成标志 |
|---|---|---|
| URDF 关节轴、零位、限位、质量和质心 | `src/rebotarm_description/urdf/robot_arm.urdf` | RViz 零位和小幅运动方向一致 |
| 电机 ID、UART 限速、MIT 增益、力矩上限、夹爪角度 | `src/rebotarm_description/config/rebotarm_hardware.yaml` | 七轴反馈 ID、单位、限幅通过实测 |
| TCP 和相机传感器/手眼外参 | `src/rebotarm_description/config/d435_mount.yaml`；相机外壳安装在主 URDF | 已知点定位误差达到项目要求 |
| 抓取夹持力和成功阈值 | `src/rebotarm_tasks/config/pick_uart.yaml` | 空夹、接触、抬升试验通过 |
| 桌面和固定障碍物 | `src/rebotarm_planning/config/planning_hardware.yaml`、`mjcf/hardware_scene.xml` | 碰撞场景与实物一致 |

`uart.gravity.joint_offset` 和 `joint_direction` 只用于重力前馈计算。当前 UART 规划路径不会自动把 MCU 坐标转换成 URDF 坐标，因此六轴 MCU 启动零位、方向必须先与 URDF 对齐；否则不要执行抓取，也不要开启重力前馈。

### 2.2 推荐标定顺序

1. **通信、反馈和槽位映射**：STM32 上电后持续返回 131 字节；确认反馈 ID 为 `[1,2,8,9,20,1,2]`（十六进制 `[0x01,0x02,0x08,0x09,0x14,0x01,0x02]`），并确认每个反馈槽位对应实体电机型号（J1～J3 DM4340P、J4/J6 DM4310、J5/J7 PA43）。七轴状态、温度、位置和单位正确，且 `feedback_ready=true`、`feedback_state=0` 后才进入下一步。
2. **零位和方向**：支撑机械臂，复位 STM32；记录 `/rebotarm/joint_states`。每次只测试一个关节、使用很小的位置变化和低限速，确认正方向与 URDF 一致。修正 URDF 原点、关节轴或固件零位映射后重新同步。
3. **机械限位和速度**：逐轴确认实际上下限，不要直接采用电机额定值。把安全速度设置为实体验证值；UART 的速度和力矩上限不能超过固件接受范围。
4. **TCP**：用量块、尖端标记或三点法测量 `gripper_end → gripper_tcp`，更新 `d435_mount.yaml` 的 `tcp.xyz_m`。当前 SDK/规划实现要求 TCP 局部旋转保持为零。
5. **D435 手眼**：相机外壳安装已来自 CAD；`d435_mount.xyz_m/rpy_rad` 表示 CAD `camera_link` 到 RealSense `camera_sensor_frame`，当前为未验证的前表面估计值。先测量镜头坐标并更新，再用 AprilTag/标定板做眼在手上手眼标定。用已知位置的目标验证 `/rebotarm/vision/color_blocks/detections`，确认 `base_link` 下的位置误差满足抓取要求。
6. **桌面和障碍物**：测量桌面高度并更新 `planning_hardware.yaml: table_top_m`；把固定障碍物和相机碰撞代理更新到 `hardware_scene.xml`。先用空场景执行无物体的低速路径检查，再接入视觉目标。
7. **夹爪开度**：记录完全打开和完全闭合时的 PA43 角度，更新硬件 YAML 的 `gripper.position_limits.open/close`；测量手指实际间距和 `gripper_joint2` 行程。
8. **夹爪限矩和接触阈值**：在软质、已固定的测试件上低速闭合，分别记录空夹、首次接触和稳定夹持的力矩。将 `gripper.torque_limit_nm` 设置为带安全余量的设备上限，将 `hardware_torque_limit_nm` 设为本任务上限，并让 `hardware_contact_torque_nm` 小于任务上限。任何一个值不清楚都保持为 `0`，任务会拒绝真机夹取。
9. **重力前馈（最后）**：先保持 `gravity_feedforward=false`，逐轴验证位置控制。确认质量、质心、零位和方向后，再在 `uart.gravity` 中逐轴填写偏置、方向和小的 `tau_scale`，支撑机械臂进行低速验证。

每一步都先在失能状态下记录结果，再进入下一步。最小检查命令如下：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# UART bridge 已启动但尚未使能时执行
ros2 topic echo /rebotarm/uart/status --once
ros2 topic echo /rebotarm/joint_states --once
ros2 topic hz /rebotarm/joint_states

# 使用真实 D435 后检查检测结果和坐标变换
ros2 topic echo /rebotarm/vision/color_blocks/detections --once
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

进入下一步的最低条件是：`rx_frames` 持续增加、`feedback_ready=true`、七个反馈块的新鲜标志为真、`feedback_state=0`，并且关节位置/温度/单位与手工记录一致。任何一项不满足都停止标定，不要用修改超时或把开关改为 `true` 绕过检查。

标定结果只写入唯一参数源：URDF 的轴和限位写 `robot_arm.urdf`，电机与 UART 写 `rebotarm_hardware.yaml`，TCP/相机外参写 `d435_mount.yaml`，任务阈值写 `pick_uart.yaml`，桌面和固定障碍物写 `planning_hardware.yaml`/`hardware_scene.xml`。每次修改后重新执行同步脚本；不要直接编辑生成的 YAML。

### 2.3 生成和检查

修改唯一参数源后执行：

```bash
python3 scripts/sync_robot_parameters.py
python3 scripts/sync_robot_parameters.py --check
colcon build --symlink-install
source install/setup.bash
```

不要手改同步生成的 `uart.yaml`、`uart_grasp.yaml`、bringup 硬件 YAML 和 MJCF 关节参数。

### 2.4 完成标定后才修改的开关

完成上面的实体测试后，才在 `pick_uart.yaml` 修改：

```yaml
hardware_calibrated: true
hardware_torque_limit_nm: <实测任务上限>
hardware_contact_torque_nm: <实测接触阈值>
```

同时在 `planning_hardware.yaml` 填写真实桌面/障碍物参数，并将：

```yaml
hardware_scene_calibrated: true
```

这两个开关只是声明标定已经完成，程序仍会继续检查反馈新鲜度、碰撞、限位、轨迹误差和视觉抬升结果。

### 2.5 标定后的完整抓取命令

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart \
  port:=/dev/serial/by-id/实际设备 \
  task_config:="$PWD/src/rebotarm_tasks/config/pick_uart.yaml" \
  planning_config:="$PWD/src/rebotarm_planning/config/planning_hardware.yaml" \
  start_camera:=true \
  gravity_feedforward:=false
```

确认反馈和视觉正常后：

```bash
./scripts/start_uart_bridge.sh enable
ros2 run rebotarm_tasks pick \
  --config "$PWD/src/rebotarm_tasks/config/pick_uart.yaml" \
  --color red --action-timeout-padding-sec 40
```

## 3. 迁移仍需要实测的项目

| 项目 | 当前代码 | 真机所需数据 |
|---|---|---|
| 坐标系/TCP | URDF 与 MJCF 的直立 `base_link`、TCP、D435 光学帧已对齐 | 编码器零位/方向与实体 TCP；旧 CAD 基座外参需转换或重新标定 |
| D435 外参 | 安装在 `gripper_end`，当前为仿真名义安装值 | 实体支架变换、手眼标定；RealSense 发布内部 optical TF，RSP 只发布安装 TF |
| 夹爪位置/力 | 共享判断，SDK/UART 使用 PA43 Nm；输出端限制总 PD 力矩 | 全开/全闭角度、实际指间距、空载摩擦、接触阈值、安全限矩；不能按额定 5 Nm直接设夹持力 |
| 工作台/场景 | 静态桌面 + RGB-D 色块代理 + 地图过期检查 | 桌面高度/范围/位置、目标尺寸；未知障碍尚需点云地图 |
| UART 第七轴 | 七轴主机转换/传输和夹爪服务 | STM32 的 ID 7 接收、状态回传、限矩和看门狗联调 |
| SDK 反馈新鲜度 | ROS 消息过期检查、缺失反馈失败 | 当前 motorbridge 缓存没有 CAN 接收时间戳；仍需驱动提供实际接收计数/时间戳或通信看门狗，不能把周期发布的缓存当新 CAN 帧 |
| 力矩/温升 | 仿真限矩及静态重力扫描，电机映射保持 DM/PA43 | 实测持姿力矩与温升、动态轨迹负载；静态结果不保证高速到达或持续运行 |

description 硬件 YAML 的 `gripper.torque_limit_nm` 是 SDK/UART 的设备上限；任务 `hardware_torque_limit_nm` 是本次抓取请求上限。两者都需填写实测安全值，实际输出取较小值。统一 UART 入口不再用任务限矩覆盖设备限矩。PA43 原始 POS_VEL 不能保证限矩，已拒绝；使用 `gripper/set` 或限矩 MIT。MIT 的零增益保持为零，不再隐式恢复默认增益。

## 4. 真机验证顺序

1. 失能检查关节编码器、方向、限位、温度、错误码及真实通信掉线检测。
2. 单关节小幅低速运动，核对实体与 URDF/RViz 一致。
3. 验证重力方向与额定力矩余量。
4. 标定 TCP、D435 外参和桌面，检查已知点的坐标误差。
5. 在空中测试夹爪开合、受阻、机械终点与总限矩。
6. 只执行观察位、高位对准和预抓取。
7. 用低速和实测安全夹持力完成完整流程，记录位置、速度、力矩和视觉结果。

上述仿真验证不包含实体电机、真实固件或实体标定。

## 5. 抓取边界与成功条件

### 5.1 一套任务、一套规划、三个设备后端

| 共享能力 | 实现 | 后端差异 |
|---|---|---|
| 完整抓取流程/视觉确认 | `tasks/pick_task.py` | 所有后端调用同一流程 |
| 开度几何/参数 | `tasks/pick_geometry.py` + `pick_*.yaml` | 数值由后端标定配置给出 |
| 夹爪等待 | `tasks/gripper.py` | MuJoCo 转 N，PA43 转 Nm，统一接触候选和稳定判定 |
| IK/路径/轨迹 | `planning/task_server.py` | 相同有界求解器；仿真用 MuJoCo FK，硬件用已对齐的 URDF/Pinocchio FK |
| 碰撞场景 | `planning/collision_planner.py` + `scene.py` | 仿真用物理场景，硬件用真实视觉和标定静态场景 |
| 设备控制 | 三个后端 | MuJoCo 步进、SDK/CAN、UART/STM32分别实现，反馈都来自测量 |
| MCP | `agent/rebotarm_mcp_server.py` | 工具注册和调用共享任务，不再复制抓取算法 |

`rs_*` 名称保留为入口兼容标识；抓取使用 DM4340P/DM4310/PA43 机械臂与夹爪模型；乒乓球使用独立的六轴球拍变体。`hardware_scene.xml` 是同一机械臂的碰撞环境，不是另一套机器人模型。

### 5.2 成功条件

```text
张开并等待实测到位
  → 高位对准 → 垂直对准 → 预抓取 → 目标复核 → 下探
  → 闭合并持续满足接触候选
  → 仿真额外验证双指物理接触
  → 硬件同步登记载荷碰撞模型
  → 第一次抬升 → 最终抬升
  → 仿真额外验证双指保持接触
  → 新 RGB-D 帧确认抬升至少 35 mm
  → 保持 2 秒 → 再用新 RGB-D 帧确认
  → 成功
```

第七电机受力只能作为接触候选：要求反馈匹配后端、力/力矩单位正确、开度稳定、超过标定接触阈值且未到全闭机械终点。空闭到位不会算夹住；受力可能来自卡滞或碰桌，最终仍须通过视觉抬升及保持。

`SetGripper.success` 表示接受命令，`reached_position` 是当时的测量，不表示已完成夹持。完成等待在共享任务层。没有左右独立触觉传感器的真机不能提供 MuJoCo 的双指碰撞证据。

### 5.3 仿真简化和实际限制

- `joint7` 是以米为单位的滑动代理；其广义力单位为 N。PA43 输出轴反馈为 Nm，传动比及摩擦仍需实测。
- 左右夹指理想同步；接触使用对齐可见指尖的薄 box，未建模完整机构回差与柔性。
- 抓取仿真场景仍沿用较宽的自碰撞屏蔽；硬件规划场景保留连杆自碰撞检查，但对零位编码量化造成的 0.05 mm 以内浅接触采用数值容差，超过该值仍会拒绝。CAD 凸包可能产生保守误拒绝，需要实际几何核验。
- 碰撞是采样检查，尚无任意点云地图、连续碰撞证明和高速动态障碍预测。
- 夹持载荷预测只用于规划，**不会焊接物体或修改物理仿真中的物体位置**。真实接触、摩擦和滑移由 MuJoCo 步进决定。
- 低速抓取接口不适合直接执行高速颠球，后者需要独立实时控制和训练环境。

### 5.4 本次重构解决的问题

1. 抓取状态机从 MCP 移入 tasks，参数从 Python 常量移到任务 YAML。
2. MuJoCo 力与 PA43 力矩明确区分，共享反馈过期、稳定、接触和机械终点判断。
3. 共享最终视觉验证；双指接触保留为仿真附加证据。
4. 规划从 MuJoCo 包迁到 planning，统一 IK/轨迹接口并同步确认硬件载荷登记。
5. SDK 接收完整六轴流目标、夹爪限制总 PD 力矩；UART 补齐七轴主机接口和实际反馈要求。
6. 用 `backend` 选择唯一设备后端，避免同名 Action 多提供者。
7. 修正 URDF 缺失的 CAD 基座固定旋转。修正前同一姿态 TCP 相差约 0.35～0.48 m；现在多姿态 TCP、相机光学帧和重力方向均已回归检查。
8. 红色物块原抬升路径接近 J4 下限，视觉保持高度在门槛附近；现在先抬升再向基座收回 25 mm，并将最终高度设为 0.24 m，保留 35 mm 成功门槛。
9. 遮挡导致近圆柱轮廓长短边比例波动，旧 1.15 方向门槛产生大幅转腕。现在 `yaw_aspect_min_ratio=1.3` 对近圆形保持方向，长条物体按夹爪 180° 对称性选择较小转角。
