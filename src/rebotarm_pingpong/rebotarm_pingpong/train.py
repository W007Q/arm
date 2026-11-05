"""在双目视觉估计和自由球动力学上训练 PPO 残差策略。"""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rebotarm_description.parameters import description_path
from .env import PingPongEnv
from .evaluation import model_fingerprint


def train(*, output, steps=200_000, seed=42, n_envs=4, config_path=None, resume=None):
    """训练 PPO，并把本次运行所用配置和版本一起保存。"""
    output = Path(output)
    if (output / "metadata.json").exists():
        raise ValueError("output already contains a run; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    # 小型 MLP 在 CPU 上通常更快；限制线程数，把并行度交给环境进程。
    torch.set_num_threads(1)
    config_file = Path(config_path or description_path()/"config/pingpong.yaml")
    shutil.copyfile(config_file, output/"pingpong.yaml")
    # 元数据用于复现和拒绝与当前观测/动作接口不兼容的旧模型。
    metadata = {"algorithm": "PPO residual XYZ paddle velocity and local roll/pitch rates", "seed": seed,
                "requested_steps": steps, "n_envs": n_envs, "model_fingerprint": model_fingerprint(config_path),
                "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                             "gymnasium": gymnasium.__version__, "sb3": stable_baselines3.__version__,
                             "torch": torch.__version__, "numpy": np.__version__},
                "observation": "43 fixed-scale stereo-estimate and encoder Cartesian features; no VecNormalize",
                "observation_version": PingPongEnv.observation_version,
                "action": "5 residual channels: vx, vy, vz, local roll rate, local pitch rate",
                "action_shape": [5], "observation_shape": [43],
                "observation_source": "stereo_rgb", "sim_only": True,
                "physical_calibrated": False}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2))

    # 训练环境使用随机化；验证环境单独运行，避免把训练回合当作成绩。
    def make_env():
        return Monitor(PingPongEnv(randomize=True, config_path=config_path))

    env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method="spawn") if n_envs > 1 else DummyVecEnv([make_env])
    env.seed(seed)
    validation = DummyVecEnv([make_env])
    validation.seed(seed+10_000)
    callback = EvalCallback(validation, best_model_save_path=str(output), log_path=str(output),
                            eval_freq=max(1, 20_000//n_envs), n_eval_episodes=5, deterministic=True)
    if resume:
        # 只有 schema、配置指纹都匹配时才允许续训。
        from .evaluation import load_policy
        policy = load_policy(resume, config_path=config_path)
        policy.set_env(env)
    else:
        policy = PPO("MlpPolicy", env, seed=seed, device="cpu", n_steps=256, batch_size=256,
                     n_epochs=10, learning_rate=3e-4, gamma=.995, gae_lambda=.95,
                     ent_coef=.001, clip_range=.15,
                     policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.5}, verbose=1)
    policy.set_logger(stable_baselines3.common.logger.configure(str(output), ["stdout", "csv"]))
    try:
        # PPO 的动作是五维残差，环境基线负责主要的击球节奏。
        policy.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=not bool(resume))
        policy.save(str(output/"final_model"))
        metadata["actual_steps"] = policy.num_timesteps
        (output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    finally:
        env.close()
        validation.close()
    return output / ("best_model.zip" if (output/"best_model.zip").exists() else "final_model.zip")
