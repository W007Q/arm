# 上位机 UART 使用说明

```text
ROS 控制器 → bridge → link → protocol → transport → UART → STM32 → CAN
```

默认 **3000000 baud、8N1、指令发送 100 Hz**。所有入口使用同一套七轴协议：发 72 字节、收 131 字节。
反馈周期由 STM32 固件决定，`te (3)` 源码为 20 ms（50 Hz）；上位机修改 `tx_hz` 不会改变反馈周期或固件 5 ms（200 Hz）的内部控制周期。
字段布局见 [协议说明](uart_protocol_zh.md)，文件职责见 [UART 包说明](../src/rebotarm_uart/README.md)。

## 修改入口

| 修改内容 | 文件 |
|---|---|
| 收发字段、缩放、帧格式 | `src/rebotarm_uart/rebotarm_uart/protocol.py` |
| 周期、目标缓存、使能和超时判断 | `src/rebotarm_uart/rebotarm_uart/link.py` 的 `UartLink` |
| ROS 话题、服务、诊断 | `src/rebotarm_uart/rebotarm_uart/bridge.py` |
| 串口读写 | `src/rebotarm_uart/rebotarm_uart/transport.py` |
| 参数源 | `src/rebotarm_description/config/rebotarm_hardware.yaml` |

修改参数源后执行 `python3 scripts/sync_robot_parameters.py`，不要手改派生 YAML。

`uart.yaml` 和 `uart_grasp.yaml` 都使用固件对应的逐轴增益，默认关闭重力前馈。
两份配置都包含七个逻辑关节；桥接器会按 `slot_joint_ids` 重排为七个 STM32 线槽位。开启模型前馈前须先标定启动零点与模型零点的关系。
`joint_ids` 是上位机逻辑关节号 `1..7`；STM32 线槽位到逻辑关节的映射为
`slot_joint_ids=[4,6,1,2,3,5,7]`。反馈块内的本地 Motor ID 为
`[1,2,8,9,20,1,2]`（十六进制 `[0x01,0x02,0x08,0x09,0x14,0x01,0x02]`），CAN Master ID 不写入 UART 帧。

## 启动

在工作空间根目录运行，串口名换成实际设备（也可以填写 `/dev/serial/by-id/...`）：

```bash
./scripts/start_uart_bridge.sh /dev/ttyCH343USB0
```

脚本自动加载 ROS 和工作区环境，启动串口桥和位置适配器，默认 3000000 波特率、关闭重力前馈。
在另一个终端的工作空间根目录发起使能：

```bash
./scripts/start_uart_bridge.sh enable
```

脚本调用 `real_uart.launch.py`。使能先发送零增益、零限速、零前馈帧，首帧写入成功即返回；取得新鲜运行反馈后，适配器用实测位置和配置增益生成保持目标，无需另开 `mit_hold`。首次发包不检查反馈，后续继续检查反馈新鲜度和故障。
上电和首次使能时应支撑好机械臂；当前低增益、无前馈配置不能保证所有姿态都能抗重力。

需要 IK/轨迹任务时，先加载 ROS 和工作区环境，改用以下入口：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_bringup grasp.launch.py \
  backend:=uart port:=/dev/ttyCH343USB0
```

两种入口不要同时启动，否则会有两个 UART bridge 或两个位置目标源。
真机保持自己的实测反馈；同一命名空间不要同时运行假驱动或 MuJoCo 权威节点。

## CH343 插上后没有串口节点

本机 CH343（USB ID `1a86:55d3`）使用 **WCH 厂商驱动**，生成 `/dev/ttyCH343USB0`。
内核 `cdc_acm` 对应 `/dev/ttyACM0`；这里按用户要求使用厂商驱动。

本次查明：驱动原先只安装在旧内核 `6.8.0-110-generic` 下，当前内核是
`6.8.0-138-generic`，没有可用的 `ch343` 模块。旧 udev 规则仍解绑 `cdc_acm`，
随后绑定不存在的 `usb_ch343`，导致两种串口节点都消失。

WCH 官方 V2.2 源码和 DKMS 配置已准备在
`artifacts/drivers/ch343/2.2-b5515b91110f/`，已为当前内核编译成功。
在工作空间根目录运行：

```bash
./scripts/install_ch343_driver.sh
ls -l /dev/ttyCH343USB0
```

脚本会请求 sudo 密码，为当前内核安装驱动并注册 DKMS，后续内核更新时自动重建
（需要新内核对应的头文件）。它备份旧 CH343 换绑规则，改用仅匹配本型号和控制接口的
规则，并设置 `dialout` 读写权限；其他型号不变。脚本不打开串口、不发送电机命令。
如果安装完成后没有节点，重新插拔 CH343 USB。

确认节点恢复后启动：

```bash
./scripts/start_uart_bridge.sh /dev/ttyCH343USB0
```

## 发送目标

默认话题 `/rebotarm/uart/mit_targets`，类型 `std_msgs/msg/Float64MultiArray`。
按 `joint1..joint6, gripper` 顺序填入 35 个数，每轴五项：

```text
[q1, dq1, kp1, kd1, tau1, ... q7, dq7, kp7, kd7, tau7]
```

`q` 为相对 STM32 启动零点的绝对 rad，`dq` 字段实际是限速值（rad/s），`tau` 为 Nm。STM32 不得对位置再次累加；当前固件仍在下位机做 5 ms 插值。
上游应持续输出目标，bridge 按 100 Hz 采样发送；`dq=0` 会暂停该轴轨迹。

也可配置 `command_source:=per_joint`，使用 `/rebotarm/joints/<joint_name>/cmd/mit`。
任一轴超过默认 300 ms 未更新，都会停发。重复相同位置会刷新保活时间。

UART 接收 **Kp≤65.535、Kd≤5**。各关节 Kp/Kd 只维护在硬件配置
`models.hybrid.overrides.joints[].MIT`；同步脚本会按 `[J4,J6,J1,J2,J3,J5,J7]`
自动生成 UART 槽位顺序。`uart.firmware` 只维护固件限速、前馈力矩上限和协议增益上限。
默认值仍需按真实负载调试。

## 使能与停止

```bash
./scripts/start_uart_bridge.sh enable
./scripts/start_uart_bridge.sh disable
```

它们分别调用 `/rebotarm/enable` 和 `/rebotarm/disable` 服务。
enable 直接发送零输出首帧，不读取启动前的 CAN 反馈，也不等待上游目标。
**首帧写入成功，服务立即返回 `success=True`，消息为 `enable command sent`。**
该结果表示指令已写入串口；七轴实际运行状态随后由反馈确认。首帧及确认阶段的
限速、Kp/Kd、前馈均为 0，位置字段 0 不会推进固件轨迹或产生位置控制力矩。

发包后在后台检查：七轴反馈全部新鲜、七轴使能、三组 flags 均报告运行及命令新鲜，
才把 `/rebotarm/uart/status` 的 `enabled` 置为 `true` 并接入正常目标。
`command_stream_active=true, enabled=false` 表示已开始发送、尚未确认全臂运行。
确认期限默认为 2 秒；超时会在状态话题报告 `post-enable feedback timeout` 并停发，
不会让已返回的 enable 服务重新变成失败。串口写失败、worker 已停止或锁存故障仍会拒绝请求。

正常目标接管前按新测量检查 `start_tolerance`（默认 0.15 rad）；运行中 CAN 反馈过期立即停发。
若适配器未接管，临时零输出目标也会在默认 300 ms 后过期停发。
`te (3)` 下位机本身仍有约 100 ms 的使能反馈保护；上位机服务立即返回不能取消该保护。

停止服务确认的是**主机已停发**。固定协议没有失能指令，当前 STM32 在最后一条有效命令后约 **500 ms** 由看门狗请求 CAN 失能；主机不能通过 72 字节帧修改这个时间。

## 重力前馈

满足以下条件后才开启：

1. 已确认安装方向、URDF 的质量、质心及末端工具/相机/负载与实物一致；
2. 已建立 MCU 启动零点到 URDF 零点的六轴偏置，并确认每轴方向；
3. 空载低速逐轴验证，确认补偿方向正确，且输出不超过 UART 力矩上限；
4. 已安装 Pinocchio/第三方 SDK，能够加载 `sdk_urdf`。

标定值维护在 `rebotarm_hardware.yaml` 的 `uart.gravity`：`joint_offset` 是 MCU 设零姿态在 URDF 中的关节角，`joint_direction` 是电机角到 URDF 角的正负方向，`tau_scale` 默认 0.1。修改后执行同步脚本。适配器会按
`q_model=q_uart*joint_direction+joint_offset` 计算模型重力，再乘方向和缩放，并按逐轴 UART 前馈力矩上限限幅。首次验证可仅给待测轴非零缩放，其余轴设为 0 并用支架支撑。

固件每次复位会重设 UART 零点，必须重新核对偏置。这组参数只校准重力计算，不会自动把规划目标、反馈和机械限位转换到模型坐标；完整规划抓取还需要统一这些坐标。

确认标定后显式开启，默认仍是关闭：

```bash
./scripts/start_uart_bridge.sh /dev/ttyCH343USB0 gravity_feedforward:=true
```

如果使用任务入口，则把同一参数传给 `grasp.launch.py`。参数在节点启动时读取，需先失能、停止节点、再带参数重启；只执行 `ros2 param set` 不会启用该模型。重力前馈开启后仍需调用 `/enable` 才会发 UART 帧。

这是位置控制上的重力前馈，位置刚度仍然存在；它不等于纯力矩拖动模式。默认力矩上限只限制前馈项，电机内部 PD 产生的总力矩仍需电机侧限制。

## 反馈与诊断

- `/rebotarm/joint_states`：七轴实测状态，以及夹爪两指的派生位置。
- `/rebotarm/joints/<joint_name>/state`：逐轴状态和温度。
- `/rebotarm/arm_status`：主机控制状态。
- `/rebotarm/uart/status`：JSON，包含三组 flags、原始状态字、反馈 ID、新鲜度、`feedback_position_rad` 原始角度和主机收发计数。`feedback_fresh=false` 时，角度只是 STM32 缓存，不能作为当前测量。

新鲜标志为 0 的反馈不作为新测量发布。固定帧没有 MCU 时间戳和执行序号，因此诊断不提供虚构的 ACK。

### RViz 执行时失能：`upstream joint command timeout`

这表示 bridge 没有及时收到完整 MIT 目标，与 UART/CAN 反馈超时是两个独立问题。`targets_fresh=true` 只说明目标流已经恢复，不会清除先前锁存的故障。诊断的 `fault_detail.command_age_sec` 记录触发故障时各轴目标的年龄。

位置适配器按 UART 的 100 Hz 顺序处理并发送目标；轨迹服务在重规划期间持续重发最后一个已检查目标。默认 `command_timeout=0.30 s`，位置适配器的上游 `target_timeout=0.35 s`。这些是主机等待时间；主机停发后，实际失能还要等待 STM32 的约 500 ms 看门狗。

更新代码或同步配置后，必须先停止旧 launch，再重新启动 UART/轨迹后端；构建不会替换已运行进程。重启会清除主机故障；若进程未重启，则先显式 `disable` 清故障，再 `enable`。重新使能成功后，在 RViz 重新 Plan，再 Execute。

### 全部红灯却报告 J3 已使能

当前 J3 的控制 ID 是 `0x14`，反馈/Master ID 是 `0x13`。如果实物仍为红灯却报告 J3 已使能，先核对下位机实际烧录的固件是否也使用 `0x14`；源码头文件改过不等于板上 ELF 已更新。旧的 `0x10` 状态位重叠说明保留在历史迁移记录中，不适用于当前配置。

### RViz 只显示相机，Execute 报轨迹超限

先查看 RViz 的 `Error retrieving file` 日志。当前 CAD 使用 `part_*.stl` 网格；引用 `base_link.STL`、`link1.STL` 的旧款 URDF 与这些网格不匹配，会导致机械臂消失，同时使 MoveIt 与执行端的关节范围不同。恢复对应 CAD 模型后，执行 `python3 scripts/sync_robot_parameters.py --check`，重新构建并重启 UART/轨迹后端和 MoveIt，再重新 Plan。

轨迹服务启动时会核对 URDF/MJCF 限位；轨迹超限错误会列出采样点、关节、目标值和允许范围。密集轨迹的最短采样时间按控制周期计算，且仍检查关节限速，避免每点累积 0.2 秒造成 MoveIt 执行超时。

### RViz Plan 成功但执行不动

先看轨迹服务的 `trajectory requested / progress / canceled / aborted` 日志：`goal_rad` 是整段终点，`target_rad` 是当前主机目标，`measured_rad` 是实际反馈，单位均为 rad。这些记录能区分碰撞拒绝、MoveIt 超时取消和电机未跟随；主机发布目标不等于 STM32 已确认执行。

真机模式用 HPP-FCL（Pinocchio 的碰撞库）三角网格复核 MuJoCo 双网格自碰撞候选。曾记录到凸包报 `link1/link5` 穿透约 0.207 mm，但 CAD 网格仍有约 9.06 mm 间隙；这是误报。`link2/link5` 此前也触发过接触中止，后经用户确认属于机械限位，已与 `link2/link4` 一同在 SRDF 和执行端允许接触；这些豁免对相应连杆对的所有姿态生效。其他碰撞检查继续保留。

真机 MoveIt 已关闭随机修正碰撞起点；当前姿态碰撞时 Plan 会直接失败。空闲心跳保持锁存位置，不再让目标随下滑的实测位置改变；失能/重新使能会清除旧目标。

`joint_states` 只包含 URDF 的六轴和两个夹指关节；夹爪电机的原始角度由 `/<命名空间>/gripper/state` 发布。MIT 适配器及保持示例分别接收两类反馈，避免 MoveIt 持续报告 `Joint 'gripper' not found`。

若目标已变化而实测不变，结合日志逐轴核对方向、机械限位及负载下的增益。当前没有自动提高 Kp/Kd，也没有开启未标定的重力前馈；仅凭规划成功或虚拟串口执行成功，不能确认真机已具备足够驱动力矩。

执行器把主机目标相对反馈的领先量限制在 `planning_hardware.yaml` 的 `max_command_lead_rad`，当前为 0.20 rad。仅看 P 项，能够施加的力矩约为 `本轴 Kp × 当前角度误差`；例如 J5 的 `Kp=2.1`、误差 0.20 rad 时约为 0.42 N·m。这不是电机总输出力矩或前馈限值。是否足以克服实际负载，需要逐轴实测；各轴 Kp/Kd 以 `rebotarm_hardware.yaml` 的 `joints[].MIT` 为准，修改后同步派生配置并重启。

“按一次 Execute 只动一点”已有对应日志：J5 多次停在约 `+0.06008 rad`，而主机当前目标约 `−0.14030 rad`；反向目标中也曾停在 `+0.18292 rad`，目标却为 `+0.38292 rad`。领先量耗尽后整条协调轨迹暂停推进，等待 4 秒仍无进展便失败；失败时撤回为实测姿态，下一次请求又从这里开始，因此看起来每次只动一点。这里只能确认 J5 未跟随，不能仅凭这些角度区分增益不足、负载/静摩擦、方向或 MCU/CAN 执行问题。

执行器已改为跨路点、逐轴检查进展，参数为 `planning_hardware.yaml → motion_stall_timeout_sec`，默认 4 秒。每个未到位轴必须累计朝目标移动至少 `0.002 rad` 才刷新自己的计时。终点不再固定只等 4 秒：只要持续有上述进展，就继续等待；六轴误差均 ≤ `0.025 rad` 且连续保持 `0.2 s` 才返回成功，并继续保持规划终点。取消、反馈过期、碰撞、失能或大跟踪误差仍会结束执行。慢速跟随与真正卡住的验证见 [执行修复验证](validation/trajectory_execution_20261103.md)。

### 使能报关节越限

`joint 3 exceeds configured limits: pos=... rad, allowed=[0, 2.0943951] rad`
表示 UART 已开始握手，但目标未通过上位机的关节限位检查；J3 的模型范围是 0～120°。

CAN 零位可能解码为约 ±0.000191 rad，而 UART 位置刻度是 0.001 rad。
主机仅接受编码后落在合法范围内的边界量化误差，并把目标对齐到限位内的 UART 刻度；原始反馈不改写，URDF 限位不放宽。
真实越限会显示具体字段、数值和范围。若偏差明显，核对启动零位和方向与 URDF 是否一致；`uart.gravity` 的偏置仅用于重力计算，不会自动修正控制坐标。

MoveIt 规划起点若只报 J3 约 `-0.00019 rad` 越过下限 `0 rad`，这是 CAN 零位量化误差。UART bridge 现在只在反馈值四舍五入到 `0.001 rad` 后落回合法边界时，将 `joint_states` 对齐到边界；超过一个量化步长的越限仍会原样保留并阻止规划。

当前验证使用 PTY 虚拟串口；实板需核对绝对位置语义、字段范围、反馈 ID 和看门狗动作。
