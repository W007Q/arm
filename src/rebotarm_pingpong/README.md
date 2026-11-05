# rebotarm_pingpong

独立六轴球拍仿真、双目 RGB 定位和 PPO 笛卡尔残差控制。训练直接调用 MuJoCo，不依赖 ROS 节点，不连接真实电机。

**球始终为自由球，可平移、旋转、侧飞和掉拍。** 历史名称“一维”仍用于兼容入口；当前动作是五维 `[vx, vy, vz, roll_rate, pitch_rate]`，控制器用双目 RGB 估计追踪横向偏差并修正拍面倾角。仿真真值只用于物理、评分和误差检查。原抓取场景及夹爪模型独立保留。

| 模块 | 用途 |
|---|---|
| `vision.py` | 两台固定相机 RGB → 白球检测 → 三维位置/速度与跟踪质量 |
| `control.py` | 笛卡尔基线、阻尼 IK、编码器 FK、名义前馈、MIT 指令延迟和限矩 |
| `env.py` | 自由薄壳球动力学、相机观测、奖励和失败判定 |
| `train.py` / `evaluation.py` | PPO、独立种子对照与模型/观测版本检查 |
| `diagnostics.py` | 落球反弹收敛及相机检查 |
| `viewer.py` / `cli.py` | 自由观察窗口及运行入口 |

参数入口：`src/rebotarm_description/config/pingpong.yaml`。XML 是生成文件，配置修改后运行 `python3 scripts/sync_robot_parameters.py`。

```bash
# 在工作空间根目录；无需启动 ROS
python3 -m pip install -r src/rebotarm_pingpong/requirements.txt
export PYTHONPATH="$PWD/src/rebotarm_description:$PWD/src/rebotarm_pingpong:$PYTHONPATH"
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli baseline --episodes 3
python3 -m rebotarm_pingpong.cli view --camera free
# 左键旋转、右键平移、滚轮缩放
# 查看器卡顿时只降低预览相机频率；训练和评估仍用配置的 60 Hz
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli view --camera free --sensor-fps 20

# 新训练示例；每次选新的输出目录
MUJOCO_GL=egl python3 -m rebotarm_pingpong.cli train --steps 200000 --envs 1 \
  --output artifacts/pingpong/my_camera_run
```

旧固定 XY / 真值观测以及 26 维观测的一维 PPO 模型已失效，加载时会被 schema 和指纹检查拒绝。短训练不代表策略收敛；球拍/球碰撞、电机与视觉参数仍需实物标定。当前没有真实相机和真机控制入口。

完整说明、评估命令和标定路线见 [乒乓球仿真与 PPO](../../docs/pingpong_zh.md)。
