# 抓取架构重构验证（2026-10-10）

## 1. 验证范围

本轮完成共享任务、共享夹爪判断、共享规划及唯一 backend 启动入口的代码重构。
所有实验使用 MuJoCo、假电机对象或 PTY 串口对端；没有连接实体电机。
未删除其他模型或开发文件。硬件碰撞场景仍引用同一个 `robot.xml`。

## 2. 构建与自动化测试

- 9 个相关包构建通过：description、tasks、planning、agent、mujoco_rs、
  controller、uart、bringup、moveit_config。
- 61 项测试通过。覆盖模型/相机坐标、重力方向、视觉、共享夹爪等待、机械终点
  误判、反馈过期、轨迹进度、碰撞、硬件地图更新、总 PD 限矩、UART 六/七轴及
  SDK 夹爪 Action 超时失败语义。
- 七轴 UART 使用 PTY 对端验证了批量和逐轴两种传输，包括第七轴及左右夹指
  状态；这不代表实际 STM32 固件或 CAN 链已经通过。
- 22 个姿态对照 URDF/Pinocchio 与 MuJoCo 的 TCP 和 D435 光学帧，容差
  `2e-7`；重力矩允许 MJCF 滑动代理质量导致的 `0.01 Nm` 差异。
- 包/场景 XML、D435 两种 TF 配置、Python 编译及启动脚本语法检查通过。

重跑测试：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH="/home/wq/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
ROS_DOMAIN_ID=165 ROS_LOG_DIR="$PWD/log/refactor_tests" MUJOCO_GL=egl \
  python3 -m pytest -q tests src/rebotarm_uart/test
```

## 3. 从统一入口执行完整抓取

使用安装后的 `grasp.launch.py backend:=mujoco` 与
`ros2 run rebotarm_tasks pick`，检查两个机械臂 Action 各只有一个提供者。
每个颜色在独立域、独立初始场景中验证。

| 物体 | 相机首次确认抬升 | 保持 2 秒后抬升 | 结果 |
|---|---:|---:|---|
| 红色方块 | 51.8 mm | 48.8 mm | 通过 |
| 蓝色长方块 | 72.3 mm | 71.0 mm | 通过 |
| 黄色圆柱 | 61.9 mm | 60.4 mm | 通过 |

三个结果都经过闭合接触候选、双指物理接触、抬升后的双指保持及新 RGB-D 帧
验证。成功门槛保持为 35 mm，没有用物理真值代替最终视觉证据。
这是固定场景功能回归，未统计随机场景成功率。

```bash
ROS_DOMAIN_ID=166 ROS_LOG_DIR="$PWD/log/unified_test" MUJOCO_GL=egl \
  python3 tests/launch_grasp_smoke.py red
# 可将 red 改成 blue 或 yellow；每次会启动和清理自己的仿真进程
```

原始实验输出：`/tmp/rebotarm_unified_pick.log`、
`/tmp/rebotarm_unified_pick_blue.log`、`/tmp/rebotarm_unified_pick_yellow2.log`。

## 4. 当前姿态静态力矩

从红色抓取实验取抓取和最终抬升的六轴目标姿态，额外载荷为场景中红色物块的
45 g，按 TCP 点质量计算；机械臂、夹爪和相机质量来自当前模型。

| 关节/电机 | 额定力矩 | 抓取姿态需求 | 抬升姿态需求 | 抓取时额定利用率 |
|---|---:|---:|---:|---:|
| J1 / DM4340P | 9 Nm | ≈0 Nm | ≈0 Nm | ≈0% |
| J2 / DM4340P | 9 Nm | 6.697 Nm | 5.980 Nm | 74.4% |
| J3 / DM4340P | 9 Nm | 4.333 Nm | 4.285 Nm | 48.1% |
| J4 / DM4310 | 3 Nm | 0.538 Nm | 0.396 Nm | 17.9% |
| J5 / PA43 | 5 Nm | 1.391 Nm | 1.390 Nm | 27.8% |
| J6 / DM4310 | 3 Nm | 0.003 Nm | 0.045 Nm | 0.1% |

这两个目标姿态的静态重力需求均低于额定力矩，最大利用率为 J2 的 74.4%。
J4 在抬升目标接近 -90° 限位：几何可达性与力矩充足是不同条件，仍须遵守 IK
容差与限位。结果不包含加速、摩擦、碰撞冲击或长期温升，也不是实机电流测量。

扫描器现在将额外载荷施加在 TCP，原来施加在手腕原点会低估或误估某些力矩。
原始输出：`/tmp/rebotarm_refactor_torque.txt`。

```bash
ros2 run rebotarm_mujoco_rs rs_gravity_scan --payload 0.045 --pose-only \
  --pose grasp=-0.335855,-1.914513,1.684191,-1.343377,0.001031,-1.911958 \
  --pose lift=-0.335330,-1.813688,1.845187,-1.570796,0.001242,-1.911529
```

## 5. 已定位的问题与真机剩余条件

1. URDF 缺失固定 CAD 基座旋转，同一姿态与仿真 TCP 相差约 0.35～0.48 m。
   现在两份模型具有同一个直立参考系；已有旧 CAD 坐标的实体标定需转换。
2. 旧红色抬升路径接近 J4 下限，保持后的视觉高度在成功门槛附近。
   现在先抬升再收回 25 mm，最终高度 0.24 m。
3. 黄色圆柱因轮廓遮挡触发大幅转腕，虽有双指接触却失去可用于验证的图像。
   现在近圆形轮廓保持方向，长条物体按夹爪 180° 对称性选择较小转角。
4. 只改 MuJoCo `geom_size` 不会更新编译后的碰撞包围盒。
   硬件单几何代理现在同步更新包围半径、AABB 和 BVH，新增碰撞回归覆盖。
5. 单独限制 feed-forward 力矩无法限制电机内部 PD；SDK/UART 的 PA43
   改为主机计算总 PD 后限矩输出，零增益不再自动替换为默认值。
6. SDK/UART 完整抓取仍需实体 TCP、D435、桌面、PA43 标定及固件联调。
   当前 motorbridge 没有实际 CAN 接收时间戳，周期 ROS 缓存发布不能证明 CAN
   反馈新鲜。这项需通过接收计数/时间戳或真实通信看门狗补齐。

完整文件职责、配置位置和启动方法见 [src_architecture_zh.md](src_architecture_zh.md)。
