# 可选的 ros2_control UART 适配层

这个包是迁移用的可选路径，任何默认 launch 都不会启动它。它提供六轴
`position` 命令接口与 `position/velocity/effort` 状态接口，供标准
`JointTrajectoryController`（JTC）使用：

```text
JTC → UartSystem → /rebotarm/arm/command/position
    → posvel_to_mit → UART worker → STM32 → CAN 电机
```

消息按 `joint1..joint6` 顺序包含绝对位置和非负限速值。只有现有 UART
worker 打开串口，原来的协议、使能服务和看门狗继续生效。硬件组件和
JTC 启动后均为 inactive；组件必须收到新鲜的六轴状态和已使能状态才能
激活，初始命令对齐实测位置。反馈中断后停止输出，恢复反馈不会自动续动。

## 离线验证

```bash
colcon build --packages-select rebotarm_msgs rebotarm_ros2_control --symlink-install
colcon test --packages-select rebotarm_ros2_control --event-handlers console_direct+
```

测试只发布合成反馈，不打开串口。覆盖无反馈拒绝激活、实测位置对齐、
六轴目标发送、反馈失联锁存以及显式重新激活。

## 单独调试该路径

先启动现有 UART bridge 和 `posvel_to_mit`，再启动本包。不要同时运行
`grasp.launch.py` 的 `task_server`；它与 JTC 会争用同一组目标。

```bash
# 终端 1：唯一的串口后端
./scripts/start_uart_bridge.sh /dev/ttyCH343USB0
# 终端 2：标准控制器，暂不激活
ros2 launch rebotarm_ros2_control uart_control.launch.py
# 终端 3：原 UART 握手完成后，再激活组件和 JTC
./scripts/start_uart_bridge.sh enable
ros2 control set_hardware_component_state RebotArmUart active -c /rebotarm/controller_manager
ros2 control set_controller_state arm_controller active -c /rebotarm/controller_manager
```

停止时先把 JTC 和组件设为 inactive，再调用现有 `disable`。反馈中断后
重新开始也需要先 inactive，再从实测状态 active。

JTC 的 Action 地址是 `/rebotarm/arm_controller/follow_joint_trajectory`。
默认 RViz 真机入口仍连接 `task_server` 的 `/rebotarm/follow_joint_trajectory`，
两者不能同时执行轨迹。可选 JTC 激活后，要使用 RViz，可另开终端显式选择它：

```bash
ros2 launch rebotarm_moveit_config hardware.launch.py trajectory_controller:=arm_controller
```

本包发布机械臂 TF，使用 UART 的实测关节状态。此路径尚未完成真实电机
执行验证，也尚未接入 Servo、夹爪控制器或 MuJoCo 的 ros2_control 插件。
它不会经过 task_server 的执行中碰撞复核、跟踪进展保护和动态抓取场景；
这些功能完成标准控制器迁移前，视觉抓取继续使用默认路径。

`read/write` 只交换缓存，ROS 订阅和发布在独立线程处理；这个过渡层仍有
ROS 调度延迟，不能据此宣称获得硬实时控制或更高的 UART 实际发送频率。
