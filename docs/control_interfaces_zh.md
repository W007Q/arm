# 控制接口与后续路线

目前以 **UART → STM32 → CAN** 为真机主线。先让 RViz 和视觉抓取共用一个低速位置接口，再逐步接入标准控制器；高速颠球单独验证感知与控制周期。

## 1. 这次统一了什么

```text
MoveIt / 抓取任务
      ↓ FollowJointTrajectory / MoveToPose
rebotarm_planning task_server（规划、碰撞检查、轨迹与保持心跳）
      ↓ /<arm_namespace>/arm/command/position
      ├─ UART：posvel_to_mit → uart_bridge → STM32 → CAN
      ├─ MuJoCo：mujoco_sync → 物理仿真
      └─ 可选 SDK：MotorPassthrough → HardwareManager
```

统一消息是 `rebotarm_msgs/ArmPositionCommand`：

| 字段 | 含义 |
|---|---|
| `stamp` | 上游 ROS 生成时间；适配器的断流超时按本机单调时钟计算 |
| `positions[6]` | 固定 `joint1..joint6` 顺序的绝对角度，单位 rad |
| `velocity_limits[6]` | 非负的位置轨迹限速，单位 rad/s；0 暂停该轴目标轨迹 |

每条消息完整更新六轴，避免逐关节回调拼出新旧混合目标。UART 会把限速降低到固件允许值；MuJoCo 会降低到其 `arm_target_speed`。两者都有独立反馈与超时判断，夹爪仍使用已有服务。

SDK 也保留配置的控制模式：MIT 模式用主机限速推进位置参考，继续使用原 Kp/Kd；POS_VEL 模式使用 SDK 组级命令。批量接口不会自动把电机切成另一种模式。

形状、有限值及非负限速校验只维护在 `rebotarm_description/command_interface.py`。限位、增益、力矩和电机映射继续从 URDF 与 `rebotarm_hardware.yaml` 派生。

旧逐关节接口只供迁移：MuJoCo 可显式设 `legacy_joint_command_topics:=true`；SDK/Fake 可设 `arm_command_source:=per_joint`。它们与批量订阅互斥。正式规划器只发布批量目标，UART 的位置适配器只接批量目标。底层 `uart/mit_targets` 是 UART 内部适配接口，位置任务不应直接发布它。

**完成的是统一位置目标层；当前默认的 task_server 仍是自定义轨迹执行器。** 独立 PPO 环境暂时不接 ROS，也没有真机颠球入口。

### RViz 与抓取是否使用相同的规划

两者共用执行和通信接口，当前的目标生成、IK 和路径规划不同：

| 入口 | 目标与 IK | 路径规划 | 后续执行 |
|---|---|---|---|
| RViz 拖动末端 | 交互标记给出目标位姿，MoveIt 使用 KDL IK 求关节角 | MoveIt 的 OMPL/RRTConnect 规划关节轨迹 | `FollowJointTrajectory → task_server → ArmPositionCommand` |
| 视觉抓取 | YOLO/D435 给出物体位置，抓取任务生成预抓取、接近、抬升等位姿；`task_server` 用有界最小二乘 IK 求关节角 | 自定义 `CollisionPlanner` 检查直线关节路径，必要时用 RRT 绕障 | 同一条轨迹执行与位置命令链 |

IK 负责“末端到这个位姿时，各关节应转到哪里”，路径规划负责“从当前位置怎样避障到达”。MoveIt 的规划流程会调用 IK；两者可以配合使用。当前默认视觉抓取入口没有调用 MoveIt，不能因为两者都在 RViz 中显示就认为规划算法相同。RViz 已生成关节轨迹后，`task_server` 不再重复求 IK，但会逐段复核碰撞并在必要时重新规划连接路径。

执行按实测反馈推进：目标相对实测值的领先量受限，缓慢运动可延长执行。终点成功条件为六轴误差均不超过 `0.025 rad`，连续保持 `0.2 s`；成功后继续保持规划终点。整条路径和终点等待共用逐轴停滞检查，任一未到位轴连续 `motion_stall_timeout_sec`（真机默认 4 秒）没有累计朝目标移动至少 `0.002 rad` 就失败，不会靠其他轴运动、切路点或持续发包延长计时。它不是“计划时间一到就停止”。

## 2. MoveIt 2 与 MoveIt Servo

MoveIt Servo 是 MoveIt 2 中的实时伺服组件，不是另一个独立规划系统。

| 项目 | MoveIt 规划执行 | MoveIt Servo |
|---|---|---|
| 输入 | 目标位姿、目标关节角或运动约束 | 连续的关节增量、笛卡尔速度等；具体接口取决于版本 |
| 输出 | 带时间的轨迹，交给控制器执行 | 短周期控制指令，交给控制器执行 |
| 碰撞 | 检查起点、终点及路径，可搜索绕障路径 | 根据当前状态检查并减速/停止，不负责搜索完整绕障路径 |
| 适合 | 抓取、准备姿态、RViz Plan/Execute | 连续遥操作、目标微调、视觉跟随 |

当前 RViz 的交互标记只改变规划目标；点 Plan 预览，点 Execute 才执行。安装 Servo 本身不会把这个标记变成连续示教器，还需要输入节点、坐标系约定和控制权切换。

如果“拖动示教”指用手拖真实机械臂，还需要低阻抗/重力补偿、合适的力矩接口和实测模型；Servo 不替代这些控制功能。

## 3. ros2_control 的迁移边界

新增的 `rebotarm_ros2_control` 是可选过渡层：`SystemInterface → ArmPositionCommand → 现有 UART worker`。它复用已有协议和看门狗，不再打开一个串口；反馈来自真实 `joint_states` 和 `arm_status`。没有新鲜反馈、没有显式使能时拒绝激活，不伪造零状态。

它提供 `/<ns>/arm_controller/follow_joint_trajectory`，默认 task_server 仍提供 `/<ns>/follow_joint_trajectory`。MoveIt 默认连接后者，显式设置 `hardware.launch.py trajectory_controller:=arm_controller` 才连接可选 JTC。可选路径的启动与限制见 [ros2_control 说明](../src/rebotarm_ros2_control/README.md)。两条执行器不能同时发布同一个位置目标话题。

下一步应先完成自动控制权切换、执行保护与动态抓取场景迁移，再接 Servo；真机验证后才替换默认路径。目前通过单独启动选择执行器，可选 JTC 不经过 task_server 的执行中保护。当前 ROS 消息中转不具备硬实时保证，也不适合作为高速颠球最终链路。

## 4. 哪些包保留，哪些可以退出主线

| 包或入口 | 当前用途 | 建议 |
|---|---|---|
| `rebotarmcontroller` | SDK/CAN 驱动、FakeDriver；部分 launch 仍依赖 | 保留为可选后端；UART 完全替代后再删依赖与包 |
| `rebotarm_mujoco_rs` | 抓取物理、模拟相机、电机模型、重力扫描 | 保留；只逐步撤下其中旧规划/视觉转发入口 |
| `rebotarm_agent` | MCP/文本调用抓取任务 | 可选，默认不启动；三项核心目标不依赖它 |
| `rebotarm_moveit_demos` | 抓放、画矩形示例 | 可移到 examples/archive，先清理引用 |
| 旧 `tasks pingpong_controller` | 正弦运动示例，不处理真实球闭环 | 退出正式颠球入口 |

本次没有删除这些包，避免破坏已有启动链。减少默认路径与重复接口，通常比直接删包更有价值。

## 5. 高速颠球必须先解决什么

现有链路适合低速抓取和示教；把 PPO 训练步数或 ROS 发布率调高不能弥补执行与视觉延迟。

1. **测量运动能力。** 当前 UART 六轴限速约 0.3/0.5 rad/s，部分轴力矩较保守。先测关节响应、负载、拍面速度及工作空间，再判断是否需要减重或换电机，不能只放宽软件限值。
2. **同步升级通信与固件。** 115200、8N1 下单帧命令至少约 6.25 ms，反馈约 11.37 ms。131 字节反馈在 100 Hz 时需要 131 kbit/s，已超过波特率。全双工应分别计算两个方向；250 Hz 可从 460800、500 Hz 可从 921600 评估，但必须同步验证 MCU、CAN 总线负载和调度。
3. **收敛控制语义。** 现固件把第二字段解释为位置轨迹限速，内部每 5 ms 插值；它不是 MIT 带符号目标速度。高速实验应明确采用位置/速度/力矩哪一种接口，配套时间戳、序号、校验、执行周期与本地看门狗，避免多层插值增加延迟。
4. **提高可用视觉速率并做预测。** 优先评估 120–240 FPS 的全局快门相机、短曝光、稳定照明和同步时间戳。双目位置改变后更新外参；估计器需预测未来接触时刻，而不是追赶延迟图像中的位置。型号与帧率应由球速和可接受误差决定。
5. **单独运行颠球闭环。** MoveIt 负责准备位、收尾和低速移动；颠球采用相机状态估计、接触预测、基线控制及 PPO 残差。目标可先按 100–250 Hz 策略、200–500 Hz 执行设计，再由端到端实测决定，不能把频率目标当成已实现能力。
6. **先辨识再迁移。** 测量恢复系数、摩擦、自旋、空气阻力、电机响应、视觉噪声和整条链路延迟，把测得的分布用于随机化。保留自由 XY、薄壳惯量和掉拍失败；这些条件仍不能证明接触模型与真机一致。

资料优先看：[MoveIt 2](https://github.com/moveit/moveit2)、[Servo 教程](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)、[ros2_control](https://control.ros.org/humble/)、[ros2_control demos](https://github.com/ros-controls/ros2_control_demos)、[UR ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)、[Franka ROS 2](https://github.com/frankaemika/franka_ros2)。前两者展示规划与伺服的分工，后三类硬件接口适合作为本项目下一步的组织参考；不要照搬其他机器人已验证的实时性指标。

## 6. 本轮验证范围（2026-10-26）

- 9 个相关 ROS 包构建通过，9 个派生参数文件一致性检查通过。
- 130 项 Python 回归通过，含批量消息、限速 0、SDK 模式保持、UART 虚拟串口和跟踪保护；2 项 C++ 合成反馈测试通过。
- 独立 ROS 域完成一次红色物体仿真抓取：双指接触、抬升约 52.7 mm，持续保持 2 秒后仍抬升约 51.1 mm。这是一次完整流程检查，不代表全部物体和随机场景的成功率。
- 补齐三个后端共用的 `gripper_state` 启动，修复开夹成功却缺少完成反馈的问题；补齐 MuJoCo 的 `gravity_bias_scale` 参数透传。
- 无串口的可选 JTC 启动检查通过，硬件组件及控制器保持 inactive；未发送真机使能或运动指令。真实电机执行、Servo 连续示教和高速颠球迁移仍待后续验证。
