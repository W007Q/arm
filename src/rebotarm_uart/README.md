# rebotarm_uart

上位机通过 **3000000 baud、8N1 UART** 控制 STM32，由 STM32 转发 CAN。
只使用七轴固定协议：以 **100 Hz** 发送 **72 字节**指令，接收 **131 字节**反馈。
反馈周期由 STM32 决定；`te (3)` 源码为 **50 Hz**，修改主机发包频率不会改变反馈周期。

## 代码怎么看

| 文件 | 职责 |
|---|---|
| `protocol.py` | 字段定义、指令打包、反馈解包、字节流拆帧 |
| `transport.py` | Linux 串口打开、读、写、关闭 |
| `link.py` | 周期收发、使能检查、限位与超时保护 |
| `bridge.py` | ROS 话题和服务映射 |
| `posvel_to_mit.py` | 可选的位置目标 → MIT 指令适配器 |
| `hold.py` | 实测位置保持示例 |

主路径：`ROS → bridge → link → protocol → transport → STM32`。
没有旧协议、自动协议切换或独立兼容入口。

位置任务统一发布 `/<arm_namespace>/arm/command/position`，消息为
`rebotarm_msgs/ArmPositionCommand`：固定 `joint1..joint6` 顺序的
`positions[6]`（绝对 rad）和 `velocity_limits[6]`（非负 rad/s）。
`posvel_to_mit` 完整校验六轴后一次更新，限速 0 暂停该轴目标轨迹。
MoveIt/抓取和可选 JTC 使用同一个接口；一次只运行一个目标发布者。

## 使用约定

- ROS 目标按 `joint1..joint6, gripper` 排列；STM32 线槽位为
  `[joint4,joint6,joint1,joint2,joint3,joint5,gripper]`，反馈本地 ID 为
  `[1,2,8,9,20,1,2]`，即 `[0x01,0x02,0x08,0x09,0x14,0x01,0x02]`。
- 位置是相对 MCU 启动零点的绝对 rad，不能再次累加；速度字段是限速值，填 0 会暂停固件位置轨迹。
- 显式调用 `/rebotarm/enable` 后直接发送零限速、零增益、零前馈首帧，写入成功即返回，不等待反馈。七轴运行确认在后台进行，通过后接入正常目标并把状态 `enabled` 置为 `true`；确认超时或运行中反馈过期仍会停发。
- `/rebotarm/disable`、目标超时、反馈异常均停止发包；实际停机由 STM32 本地看门狗完成。
- 不变的位置也要持续发送。重力补偿需要持续计算并发送前馈力矩。
- 协议没有 CRC、命令序号、时间戳或独立使能帧，诊断只报告真实反馈和主机计数。

## 参数与文档

`config/uart.yaml` 和 `config/uart_grasp.yaml` 都采用硬件 YAML 的关节 MIT 增益，按 STM32 槽位自动重排，默认关闭重力前馈，均为七轴固定协议。MCU 启动零点与 URDF 标定完成后才能开启模型前馈。

在工作空间根目录启动，串口名换成实际设备：

```bash
./scripts/start_uart_bridge.sh /dev/ttyCH343USB0
# 另一终端发起使能，首帧发送成功即返回；实际运行状态查看 enabled：
./scripts/start_uart_bridge.sh enable
# 停止时：
./scripts/start_uart_bridge.sh disable
```

脚本自动加载环境，调用 `real_uart.launch.py` 启动串口桥和位置适配器。标定完成后才在启动命令末尾加 `gravity_feedforward:=true`。偏置、方向和缩放在统一硬件配置的 `uart.gravity` 中维护。
此入口的适配器已经产生初始目标，不要同时启动 `mit_hold`。
参数从 `rebotarm_description/config/rebotarm_hardware.yaml` 和 URDF 派生：

```bash
python3 scripts/sync_robot_parameters.py
python3 scripts/sync_robot_parameters.py --check
```

使用见 [上位机说明](../../docs/uart_host_guide_zh.md)，字段见 [协议说明](../../docs/uart_protocol_zh.md)。
STM32 固件不在本仓库；当前验证使用虚拟串口，尚未完成真实电机联调。
