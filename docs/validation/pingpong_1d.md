# 自由球与相机观测验证

日期：2026-10-14。仅为仿真验证，未连接实体相机或电机。文件名保留历史入口；当前控制动作已扩展为五维笛卡尔残差。

## 1. 本次改变

- 两个乒乓球场景的球都为自由关节：XYZ 与旋转均可变化，没有水平回正或碰撞后的速度重设。
- 控制和 PPO 使用双目 **RGB → 检测 → 三角测量 → 时间戳估计**；无效视觉回位，超时失败，不能回退真值。
- 40 mm、2.7 g 球按空心薄壳计算惯量：每轴 **7.2×10⁻⁷ kg·m²**。碰撞几何仍为刚性球。
- 加入完整三维阻力、旋转升力、旋转衰减与初始水平/旋转扰动；掉拍、侧向逃逸、掉落均可失败。
- 电机使用硬件 MIT 增益，共享 UART 250 Hz 发包、本地 500 Hz 编码器 PD、指令延迟、力矩响应及速度相关限矩。
- 成功必须坚持完整 15 秒且至少 20 次分离反弹；达到次数后再掉拍仍失败。

仿真真值仅用于初始化、物理、奖励、失败判定及独立误差评分，未进入球定位、基线或 PPO 观测。

## 2. 碰撞数值复核

测试装置是**完全固定的球拍和自由球**，不使用机械臂伺服或图像渲染。每个高度检查 0.125 / 0.0625 / 0.03125 ms 三种步长。

| 落球高度 | 名义恢复系数 | 接触时间 | 最大几何重叠 |
|---|---:|---:|---:|
| 0.10 m | 0.746 | 0.875 ms | 0.354 mm |
| 0.15 m | 0.752 | 0.875 ms | 0.437 mm |
| 0.20 m | 0.756 | 0.875 ms | 0.504 mm |

四分之一步长下，恢复系数最大相对变化 **1.28%**；整体最大重叠约 0.505 mm，占球半径 2.53%。偏心、带旋转和倾斜拍面的碰撞能量检查通过，越过拍面边界的落球没有反弹。

配置中的 3 ms 是 MuJoCo 约束时间常数，不是实测接触时长。当前 `calibration.status: unmeasured`：这些结果仅建立数值收敛与耗散证据。录入三个不同高度的实物数据后，诊断还会检查恢复系数、反弹高度和接触时间的拟合误差；仅填写 `measured` 无法使验证通过。

数据：[contact.json](../../artifacts/pingpong/camera_free_audit/contact.json)。

## 3. 相机与误差

两台固定外部相机位置为 `(0.65, -0.75, 0.70)` 和 `(0.65, 0.55, 0.70) m`，朝向 `(0.30, 0, 0.48) m`。观测图像为 320×240、60 FPS，包含 2 ms 曝光积分、12 ms 名义延迟、噪声、丢帧及相机标定误差先验。

初始姿态下，31 个竖直高度点全部位于双相机视野中，约 93.55% 无遮挡，球直径至少 12.13 px。这不是全工作区、全姿态可见性的保证。真实接入使用同步、去畸变的 RGB 和实测标定。

名义基线的 10 个独立回合：自由飞行阶段位置 RMSE 平均 **2.81 mm**、速度 **0.112 m/s**；全轨迹含反弹阶段为 **15.81 mm / 0.938 m/s**，平均帧龄 **21.41 ms**。飞行指标排除碰撞及反弹后 80 ms，仅用于离线诊断。反弹带来的速度突变在下一帧送达前不可见，估计没有利用碰撞真值修正。

相机数据：[camera.json](../../artifacts/pingpong/camera_free_audit/camera.json)。

![自由球场景](../../artifacts/pingpong/camera_free_scene.png)

## 4. 笛卡尔基线与 PPO 接口

当前观测为 43 维，动作为 `[vx, vy, vz, roll_rate, pitch_rate]` 五维残差。基线从双目 RGB 估计预测落点，实际横移拍面并调整局部倾角；球的 XY 和旋转仍完全自由。旧 `ppo_1d_final`、`ppo_camera_free_v2` 使用锁 XY、真值观测或旧 schema，加载检查会拒绝，不用于当前环境。

5D PPO smoke 使用最终 UART 节拍与相机观测运行 20,480 步；仅证明训练、保存和加载流程完整，不能视为收敛结果。独立评价先使用零 PPO 残差的相机基线，隔离横向控制本身。

每组用相同的独立种子 1000～1009，成功定义为完整 15 秒且至少 20 次有效反弹。

| 环境 | 控制器 | 成功回合 | 平均反弹 | 横向误差 RMS |
|---|---|---:|---:|---:|
| 名义 | 相机笛卡尔基线 | 10/10 | 45.1 | 5.53 mm |
| 随机化 | 相机笛卡尔基线 | 9/10 | 41.7 | 11.70 mm |

名义 10 个回合均运行到 15 秒；随机化有 1 个回合在关节限位结束，最少 27 次反弹。该基线已经能用真实拍面横移和倾角修正侧向误差，但随机化下仍未达到每个种子都稳定的目标。PPO 需要在此 5D 接口上重新训练和独立评估，不能把 smoke 结果当成策略性能。

模型：[best_model.zip](../../artifacts/pingpong/ppo_camera_free_v3/best_model.zip) · 训练配置/版本/指纹：[metadata.json](../../artifacts/pingpong/ppo_camera_free_v3/metadata.json) · 四组评价与轨迹：[evaluation.json](../../artifacts/pingpong/camera_free_eval_v3/evaluation.json)。旧 v1/v2 试验已随控制实现变更失效，当前加载检查会拒绝它们。

![相机估计、自由水平运动与延迟](../../artifacts/pingpong/camera_free_eval_v3/camera_trace.png)

图中的仿真参考仅用于离线评分；控制器使用 RGB 估计。

## 5. 回归与复现

- 乒乓球相关测试 **27 项通过**：包括自由球/薄壳惯量、视觉遮挡与延迟、禁止读取球真值、5D Cartesian/IK 限制、边缘碰撞失败判定、碰撞收敛及旧 checkpoint schema 拒绝。完整工作空间的 ROS 回归需在已 source 的 ROS 环境中运行。
- `rebotarm_description`、`rebotarm_pingpong` 两个包构建通过；九个派生参数文件检查一致。
- 原 URDF、抓取模型/场景、硬件 YAML 和腕部 D435 YAML 的六个保护文件 SHA-256 未改变，记录见 [protected_files.json](../../artifacts/pingpong/camera_free_audit/protected_files.json)。
- 完整红块抓取回归通过：唯一机械臂 Action 提供者，视觉抬升 61.4 mm，保持 2 秒后仍抬升 59.4 mm。

```bash
export PYTHONPATH="$PWD/src/rebotarm_description:$PWD/src/rebotarm_pingpong:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/sync_robot_parameters.py --check
MUJOCO_GL=egl python3 -m pytest -q tests src/rebotarm_uart/test
python3 -m rebotarm_pingpong.cli view --camera free
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli evaluate \
  --model artifacts/pingpong/ppo_camera_free_v3/best_model.zip \
  --episodes 10 --seed 1000 --output artifacts/pingpong/my_recheck --plot
```

交互窗口默认自由相机：左键旋转、右键平移、滚轮缩放；`--camera tracking|tracking_side|overview` 预览固定相机。

## 6. 尚需实测

球拍/安装件与连杆质量惯量、胶皮摩擦与接触、旋转气动力、相机标定/曝光/延迟、电机响应与持续负载仍需测量。四组评价的最大 RMS 力矩约为对应额定力矩的 **1.34 倍**，目前没有热模型，也不能据此判断真机可以持续运行。

随机化 PPO 评价中实际最高关节速度约 **2.38 rad/s**；主机的 2 rad/s 目标限制不等于真实运动速度的硬约束。

当前可上下颠球，但没有横向接球和倾角纠偏；自由球评价中的漏接反映这项真实能力限制。实现与标定步骤见 [乒乓球仿真与 PPO](../pingpong_zh.md)。

实验产物位于工作空间 `artifacts/pingpong/`，不纳入 Git；迁移到其他机器时需单独复制。
