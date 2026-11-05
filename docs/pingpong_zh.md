# 乒乓球仿真与 PPO

## 1. 当前实现

```text
六轴机械臂＋固定球拍＋自由球
  → 两台固定相机采集 RGB
  → 检测球、三角化位置、估计速度
  → 相机横向追踪与竖直击球基线＋PPO 笛卡尔残差修正
  → 延迟与限矩电机模型
  → 有效反弹、侧飞、掉拍和跟踪失败统计
```

入口为 `rebotarm_pingpong`。模型移除第七电机、夹爪和腕部相机，安装固定球拍；原抓取模型、参数和场景继续使用自己的文件。

**“一维”是历史入口名称。球的位置和旋转始终自由**，没有 XY 约束、回正力、碰撞后重设速度或托球装置。当前动作空间为五维 `[vx, vy, vz, roll_rate, pitch_rate]`：基线用相机预测落点追踪 XYZ，并用有限倾角反馈抑制横向漂移；PPO 在此基础上学习残差。球仍可侧飞、偏离拍面和掉落，都会造成失败。

球的控制输入来自渲染的双目 RGB 图像；名义环境与随机化环境都使用这条链路。仿真真值只用于初始状态、物理计算、奖励、失败判定及独立误差统计，不进入球状态观测或击球基线。

## 2. 文件与参数

| 文件 | 用途 |
|---|---|
| `src/rebotarm_description/config/pingpong.yaml` | 球拍、球、接触、相机、控制及实验配置 |
| `src/rebotarm_description/mjcf/robot_pingpong.xml` | 从共享机械臂生成的独立六轴球拍模型 |
| `src/rebotarm_description/mjcf/pingpong_3d_scene.xml` | 当前训练和查看使用的自由球场景 |
| `src/rebotarm_description/mjcf/pingpong_1d_scene.xml` | 保留文件入口，同样使用自由球，不再锁定 XY |
| `src/rebotarm_pingpong/rebotarm_pingpong/vision.py` | RGB 球检测、双目定位、时间戳和速度估计 |
| `src/rebotarm_pingpong/rebotarm_pingpong/control.py` | 笛卡尔基线、阻尼 IK、编码器 FK、名义前馈与电机响应 |
| `src/rebotarm_pingpong/rebotarm_pingpong/env.py` | Gymnasium 环境、自由球气动力与评价 |

六轴运动学、质量和限位继承共享 URDF，电机参数读取 `rebotarm_hardware.yaml`。乒乓球工具不继承抓取 TCP 和 D435 安装参数。修改配置后同步生成文件：

```bash
python3 scripts/sync_robot_parameters.py
python3 scripts/sync_robot_parameters.py --check
```

球直径 40 mm、质量 2.7 g，惯量明确按刚性空心薄壳计算：每轴 `I = 2mr²/3 = 7.2×10⁻⁷ kg·m²`。球有三个平移及三个旋转自由度；模型包括三维空气阻力、旋转升力、旋转衰减与风。气动力系数尚未实测。

接触使用 MuJoCo 软接触求解，加入滑动、扭转和滚动摩擦。配置中的 `contact_time_s` 是求解器时间常数，**不是实测接触时长**；阻尼比也不是恢复系数。当前步长为 0.125 ms，接触参数需要同时通过数值收敛和实物落球标定。

当前固定拍面数值落球检查（0.10～0.20 m）测得恢复系数约 0.746～0.756、接触约 0.875 ms、最大几何重叠约 0.35～0.50 mm；细化到四分之一步长后，恢复系数最大变化约 1.28%。这些结果说明数值模型比旧接触更硬且已检查步长敏感度，不能证明与实际球拍一致。

## 3. 相机安装与定位

选择**两台固定在工作区两侧的外部相机**，各自标定到机械臂基座。当前初始布局：

| 相机 | 基座坐标位置 / m | 朝向点 / m |
|---|---|---|
| `tracking` | `(0.65, -0.75, 0.70)` | `(0.30, 0, 0.48)` |
| `tracking_side` | `(0.65, 0.55, 0.70)` | `(0.30, 0, 0.48)` |

侧视能观察竖直运动，两个视角提供三维定位；固定安装避免腕部运动不断改变相机外参。安装时检查整个球拍行程和球的侧向活动区，避免手臂、拍面及支架同时挡住两台相机。

当前观测图像为 **320×240、60 FPS**，加入 2 ms 曝光积分、12 ms 名义延迟、延迟抖动、丢帧、图像噪声和标定误差。处理过程为白球候选检测、已标定双目射线三角化、速度估计，再按采集时间预测到控制时刻；不读取模拟深度、分割标签或球体坐标。遮挡或候选不匹配时无法得到新的位置，只能短时预测；超时标记无效并停止击球修正，持续丢失跟踪会结束回合。

球高度 15 cm 时，近击球速度约为 1.7 m/s，60 FPS 每帧可移动约 29 mm。真机优先选择可同步的高速全局快门相机、短曝光和充足照明，先录像测误差与延迟。现有 D435 的单机 RGB-D 抓取入口不会自动提供这套双相机能力；实际帧率、同步、标定、图像传输和球检测还需要实机接入验证。

查看窗口使用自由相机时可拖动；交互视角不会改变控制使用的两台固定相机。

## 4. 控制与 PPO

| 项目 | 当前设置 |
|---|---|
| 底层控制 / PPO | 500 Hz / 100 Hz |
| 主机指令传输 | UART 配置的 100 Hz；两次传输间保持最近的绝对目标 |
| 目标顶点 | 高于中位接触球心 15 cm |
| 拍面行程 | 中位上下各 45 mm |
| PPO 动作 | 五维 XYZ 速度与局部滚转/俯仰速度残差；受工作区、加速度、关节速度和限矩约束 |
| PPO 观测 | 相机估计的相对 XYZ/速度、拍面位姿/速度/法向、基线命令、六轴编码器、前次动作及跟踪质量，共 43 维 |
| 电机响应 | 名义传输延迟 4 ms、一阶力矩响应 4 ms、速度相关限矩 |
| 回合成功 | 坚持完整 15 秒且至少 20 次有效反弹；期间掉拍或跟踪丢失即失败 |

基线用相机估计的球位置和速度预测击球时刻，在拍面局部坐标中追踪落点，并用小幅法向反馈把横向速度和位置拉回工作区；竖直分量仍按下落速度选择向上拍速。PPO 学习五维小幅修正。相机观测无效时，球拍回位并停用击球修正，不回退球真值。

拍面位置/速度由六轴编码器和独立名义模型计算；前馈也使用该模型，不能读取实时 plant 的球状态、接触或随随机化变化的惯量。按硬件 MIT 控制模式，位置/速度/前馈指令经过传输延迟后，电机用当前编码器做本地 PD，再经过有限力矩响应和限矩。参考速度同时遵守 URDF、电机和 UART 上限；当前 UART 上限为 2 rad/s，因此配置的 0.60 m/s 拍速不保证可达。

主机发包频率直接读取硬件 YAML 的 `uart.transport.tx_hz`，固定七轴 UART 当前为 100 Hz；本地 PD 的 500 Hz 不会提高主机传输频率。修改该参数也会改变新建颠球仿真环境的传输周期，原先按 50 Hz 训练的策略需要重新评估。模拟默认采用 UART 链路，实际切换到 SDK/CAN 链路时需要验证对应时序并重新训练。

默认 PD 增益使用硬件配置中的 MIT 值。参考速度限制不意味着实际速度被强行截断；实际关节响应仍由物理积分决定，需要在评价记录中复核。峰值限矩也不能代替额定持续负载或热测试。

奖励关注脱离拍面的有效反弹、目标顶点、存活和动作平滑；持续托球不能计为反弹。球错过实际运动拍面、掉落、飞出工作区、持续接触、关节越界或长时间丢失视觉跟踪都会失败。

**旧的 `ppo_1d_final` 策略和成功率只对应固定 XY、真值观测的旧环境。** 新环境的动力学、相机观测和输入维度已经改变，需要重新训练；不应通过跳过配置检查继续使用旧模型。

## 5. 运行

在工作空间根目录执行。训练直接调用 MuJoCo，ROS 用于安装入口，程序不会连接真实电机。

```bash
python3 -m pip install -r src/rebotarm_pingpong/requirements.txt
export PYTHONPATH="$PWD/src/rebotarm_description:$PWD/src/rebotarm_pingpong:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/sync_robot_parameters.py --check

# 碰撞收敛与相机检查；无桌面时使用 EGL 渲染
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli drop
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli camera
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli baseline --episodes 10

# 自由视角：左键旋转、右键平移、滚轮缩放
python3 -m rebotarm_pingpong.cli view --camera free
# 预览卡顿时只降低查看器相机采样；训练和评估仍用 60 Hz 配置
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli view --camera free --sensor-fps 20

# 新模型训练示例；每次选择新的输出目录
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli train --steps 200000 --envs 1 \
  --seed 42 --output artifacts/pingpong/my_camera_run

# 训练后评估与查看；若产生 best_model.zip，也可评估该 checkpoint
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli evaluate \
  --model artifacts/pingpong/my_camera_run/final_model.zip \
  --episodes 10 --seed 1000 --output artifacts/pingpong/camera_free_eval --plot
python3 -m rebotarm_pingpong.cli view \
  --model artifacts/pingpong/my_camera_run/final_model.zip
```

`--config /path/pingpong.yaml` 可在内存中生成实验模型，不覆盖原抓取场景。模型、配置、硬件和观测版本变更会使 checkpoint 失效。构建后也可以 `ros2 run rebotarm_pingpong pingpong ...` 使用相同入口。

每步观测包含实际渲染，训练耗时明显高于旧的真值环境。短训练只能证明流程可运行；是否学会稳定颠球，需要独立种子的自由球评估。除成功率、顶点和反弹次数，也要查看视觉位置/速度误差、有效跟踪比例、帧龄、失败原因和关节速度/力矩。

### 查看器卡顿的来源

当前正式传感器设置为双相机 320×240、60 FPS。查看器每个策略步还要推进 80 个 0.125 ms 物理步，并在相机采样时完成两路 RGB 渲染、白球检测、模糊和噪声处理。EGL 基准中，优化前每 10 ms 仿真约需 22.8 ms 墙钟时间，其中相机管线约占 70%；优化接触扫描、三维气动标量计算和 OpenCV 阈值检测后约为 11.6–13.8 ms（随机器负载变化）。普通 CPU 因此无法稳定追上 100 Hz，窗口会出现延迟或跳帧。

卡顿主要来自相机图像处理和高频仿真与 GUI 刷新争用资源，不是相机外参本身。查看器现在用独立的显示数据快照，避免在物理和视觉计算期间长时间占用 GUI 锁；`view --sensor-fps 20` 的预览基准约 8.5–9.1 ms/策略步，能进一步改善拖动。这个参数只适合交互查看，正式训练、评估和真机迁移仍使用配置中的 60 FPS。降低预览帧率会改变观测新鲜度，不能用来宣称正式策略性能。

当前验证结果见 [乒乓球验证](validation/pingpong_1d.md)。

## 6. 接近实物的标定路线

当前 `calibration.status` 为 `unmeasured`。解除约束和改成相机观测已消除主要人为优势，但无法代替实物测量。

| 测量 | 方法及用途 |
|---|---|
| 球拍与安装件 | 称重，测安装位姿、重心与惯量；核验工具以及 `link6` 的质量 |
| 法向碰撞 | 固定实拍，从 0.10、0.15、0.20 m 落球，用高速录像测入/出速度、反弹高度和接触时长，填入 `calibration.measurements` |
| 切向碰撞 | 倾斜拍面及偏心击球，记录侧向速度和旋转；拟合摩擦及旋转响应 |
| 飞行气动力 | 录像记录自由飞行和不同初始旋转，拟合阻力、旋转升力及衰减 |
| 视觉 | 标定内外参，核验球位置与速度误差、曝光拖影、同步误差、遮挡、丢帧和端到端延迟 |
| 执行器 | 记录指令与编码器响应，测通信延迟、响应时间、速度/力矩及持续负载能力，再更新先验 |

完成测量后先拟合参数，再复核多步长收敛、录像重放和独立落球实验；随后重新训练。真机由低速拍面跟踪、单次击球逐步推进到连续颠球。横移与拍面倾角控制需要扩展动作空间，并在保持自由球和相机观测的环境中训练。

### 仍然存在的近似

- 球是**刚性薄壳惯量＋球形碰撞几何**，没有球壳/胶皮的真实形变模型；接触和气动力尚未标定。
- 圆形球拍总质量 205 g、安装距离 130 mm 是设计值；共享 URDF 的 `link6` 仍为 1 kg 和简化惯量，必须核验。
- 机械臂使用简化碰撞几何，当前忽略臂体自碰撞；模型没有完整机构回差、弹性和驱动热模型。
- 名义控制模型仍继承共享模型参数；电机延迟、响应和线性速度–力矩包络未经过实测，制动能力也需验证。
- 白球检测器面向干净背景，已加入有限曝光和标定误差先验；镜头畸变、滚动快门及复杂背景尚未建模。真实图像需同步、去畸变并实测检测误差。

## 7. 资料

- [MuJoCo 接触建模](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [Gymnasium 自定义环境](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)
- [Sim-to-Real Transfer with Dynamics Randomization](https://arxiv.org/abs/1710.06537)
- [Achieving Human Level Competitive Robot Table Tennis](https://arxiv.org/abs/2408.03906)
