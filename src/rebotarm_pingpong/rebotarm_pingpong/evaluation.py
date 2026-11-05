"""成对评估基线与 PPO，并记录独立于 PPO 奖励的物理指标。"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rebotarm_description.parameters import description_path
from .env import PingPongEnv


def model_fingerprint(config_path=None):
    """计算模型、配置和控制代码指纹，防止误用旧 checkpoint。"""
    root = description_path()
    files = [root / "mjcf/robot_pingpong.xml", root / "mjcf/pingpong_3d_scene.xml",
             Path(config_path or root / "config/pingpong.yaml"), root / "config/rebotarm_hardware.yaml",
             *[Path(__file__).parent / name for name in ("env.py", "vision.py", "control.py")]]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_policy(path, *, config_path=None, allow_mismatch=False):
    """加载并检查 43 维观测、5 维动作的 PPO 模型。"""
    path = Path(path)
    metadata = path.parent / "metadata.json"
    if not metadata.is_file() and not allow_mismatch:
        raise ValueError("checkpoint has no metadata.json; use --allow-model-mismatch only for an intentional experiment")
    if metadata.is_file() and not allow_mismatch:
        saved = json.loads(metadata.read_text())["model_fingerprint"]
        if saved != model_fingerprint(config_path):
            raise ValueError("robot/config changed since training; retrain or explicitly pass --allow-model-mismatch")
    from stable_baselines3 import PPO
    policy = PPO.load(str(path), device="cpu")
    if policy.observation_space.shape != (43,) or policy.action_space.shape != (5,):
        raise ValueError("checkpoint observation/action schema is incompatible with stereo Cartesian control; retrain")
    return policy


def evaluate(policy=None, *, episodes=10, seed=1000, randomize=False, config_path=None,
             trace_path=None, env=None):
    """运行独立回合；``policy=None`` 时只测试相机笛卡尔规则基线。"""
    owned = env is None
    env = env or PingPongEnv(randomize=randomize, config_path=config_path)
    rows, traces = [], []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed+episode)
        total_reward = 0.
        while True:
            # 零动作表示不增加 PPO 残差，保留规则基线的真实横向控制。
            action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if trace_path and episode == 0:
                position, velocity = env.truth_ball_state()
                estimate = env.observer.estimate
                paddle, paddle_velocity, paddle_rotation = env.servo.paddle_state(env.data)
                # CSV 继续保留旧 residual 列（即 Z 残差），再追加 vx、vy、
                # 横滚/俯仰速度和拍面状态，便于确认横向纠偏是否真的发生。
                z, vz = float(position[2]), float(velocity[2])
                residual = float(np.asarray(action).reshape(-1)[2])
                true_lateral = position[:2] - paddle[:2]
                traces.append([env.data.time, z, vz, paddle[2], paddle_velocity[2], residual,
                               *np.asarray(action).reshape(-1)[:2],
                               *np.asarray(action).reshape(-1)[3:5], info["bounces"],
                               *env.data.qpos[:6], *env.data.ctrl, *position[:2], *velocity[:2],
                               *paddle, *paddle_velocity, *paddle_rotation[:, 2], *true_lateral,
                               float(np.linalg.norm(true_lateral)), *estimate.position, *estimate.velocity,
                               int(estimate.valid), estimate.confidence, estimate.age_s])
            if terminated or truncated:
                break
        heights = np.array(info.pop("apex_heights_m"))
        target = env.config["scene"]["target_apex_m"]
        info.update(seed=seed+episode, reward=total_reward,
                    apex_mae_m=float(np.mean(abs(heights-target))) if len(heights) else None,
                    mean_apex_m=float(np.mean(heights)) if len(heights) else None)
        rows.append(info)
    errors = [r["apex_mae_m"] for r in rows if r["apex_mae_m"] is not None]
    summary = {"episodes": episodes, "seed_start": seed, "randomized": randomize,
               "success_rate": float(np.mean([r["success"] for r in rows])),
               "mean_bounces": float(np.mean([r["bounces"] for r in rows])),
               "min_bounces": min(r["bounces"] for r in rows),
               "mean_apex_mae_m": float(np.mean(errors)) if errors else None,
               "mean_reward": float(np.mean([r["reward"] for r in rows])),
               "terminations": dict(Counter(r["termination"] for r in rows)),
               "max_joint_speed_rad_s": max(r["max_joint_speed_rad_s"] for r in rows),
               "max_torque_ratio": max(r["max_torque_ratio"] for r in rows),
               "observation_source": "stereo_rgb",
               "observation_version": env.observation_version,
               "physical_calibrated": False,
               "max_rms_rated_torque_ratio": max(r["max_rms_rated_torque_ratio"] for r in rows)}
    for key in ("horizontal_rmse_m", "bounce_radial_rmse_m", "max_bounce_radial_error_m",
                "vision_position_rmse_m", "vision_velocity_rmse_m_s", "flight_position_rmse_m",
                "flight_velocity_rmse_m_s", "vision_valid_fraction", "camera_missed_fraction", "mean_frame_age_s"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    if trace_path:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "ball_z_m", "ball_vz_m_s", "paddle_z_m", "paddle_vz_m_s", "residual",
                             "action_vx", "action_vy", "action_roll_rate", "action_pitch_rate", "bounces",
                             *[f"q{i}_rad" for i in range(1, 7)], *[f"tau{i}_nm" for i in range(1, 7)],
                             "ball_x_m", "ball_y_m", "ball_vx_m_s", "ball_vy_m_s",
                             "paddle_x_m", "paddle_y_m", "paddle_z_fk_m",
                             "paddle_vx_m_s", "paddle_vy_m_s", "paddle_vz_fk_m_s",
                             "paddle_normal_x", "paddle_normal_y", "paddle_normal_z",
                             "true_lateral_error_x_m", "true_lateral_error_y_m", "true_lateral_error_m",
                             "estimated_x_m", "estimated_y_m", "estimated_z_m",
                             "estimated_vx_m_s", "estimated_vy_m_s", "estimated_vz_m_s",
                             "track_valid", "track_confidence", "frame_age_s"])
            writer.writerows(traces)
    if owned:
        env.close()
    return {"summary": summary, "episodes": rows}


def compare(policy, *, episodes=10, seed=1000, config_path=None, output=None):
    """在名义和随机化环境中分别比较规则基线与 PPO。"""
    results = {}
    for randomize in (False, True):
        name = "randomized" if randomize else "nominal"
        for label, controller in (("baseline", None), ("ppo", policy)):
            trace = Path(output)/f"{name}_{label}.csv" if output else None
            results[f"{name}_{label}"] = evaluate(controller, episodes=episodes, seed=seed,
                                                 randomize=randomize, config_path=config_path, trace_path=trace)
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/"evaluation.json").write_text(json.dumps(results, indent=2))
    return results
