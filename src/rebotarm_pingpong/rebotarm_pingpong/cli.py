"""命令行入口：python -m rebotarm_pingpong.cli {baseline,train,evaluate,view,drop,camera}。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main(argv=None):
    """命令行入口：基线、训练、评估、查看和物理/相机诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "evaluate", "train", "view", "drop", "camera"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="pingpong YAML override (in-memory generated model)")
        if name in ("baseline", "evaluate"):
            p.add_argument("--episodes", type=int, default=10)
            p.add_argument("--seed", type=int, default=1000)
            p.add_argument("--output", default=f"artifacts/pingpong/{name}")
        if name in ("evaluate", "view"):
            p.add_argument("--model", required=name == "evaluate")
            p.add_argument("--allow-model-mismatch", action="store_true")
        if name == "evaluate":
            p.add_argument("--plot", action="store_true", help="export comparison.png with physical metrics")
        if name == "baseline":
            p.add_argument("--randomize", action="store_true")
        if name == "train":
            p.add_argument("--steps", type=int, default=200_000)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--envs", type=int, default=4)
            p.add_argument("--output", required=True)
            p.add_argument("--resume")
        if name == "view":
            p.add_argument("--seed", type=int, default=1000)
            p.add_argument("--randomize", action="store_true")
            p.add_argument("--snapshot", help="save a frame headlessly instead of opening GUI")
            p.add_argument("--camera", choices=("free", "overview", "tracking", "tracking_side"), default="free",
                           help="GUI camera: free allows mouse rotation/pan/zoom; named cameras have fixed poses")
            p.add_argument("--sensor-fps", type=float, default=None,
                           help=("override simulated RGB capture rate for this viewer only; "
                                 "20-30 Hz reduces camera rendering load (default: config value)"))
        if name in ("drop", "camera"):
            p.add_argument("--output", default=f"artifacts/pingpong/{name}.json")
    args = parser.parse_args(argv)
    # 无桌面时使用 EGL；必须在导入 MuJoCo 前设置图形后端。
    if args.command != "view" or args.snapshot:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if hasattr(args, "episodes") and args.episodes < 1:
        parser.error("episodes must be positive")
    if args.command == "view" and args.sensor_fps is not None and (not np.isfinite(args.sensor_fps) or args.sensor_fps <= 0):
        parser.error("sensor-fps must be finite and positive")
    if args.command == "train":
        # 训练目录会保存模型、配置和元数据，便于之后复现实验。
        if args.steps < 1 or args.envs < 1:
            parser.error("steps and envs must be positive")
        from .train import train
        checkpoint = train(output=args.output, steps=args.steps, seed=args.seed, n_envs=args.envs,
                           config_path=args.config, resume=args.resume)
        print(f"Saved policy: {checkpoint}")
        return
    if args.command in ("drop", "camera"):
        # 诊断命令只检查接触数值或相机覆盖，不启动 Gym 回合。
        from .diagnostics import camera_report, drop_report
        report = (drop_report if args.command == "drop" else camera_report)(config_path=args.config)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    from .evaluation import compare, evaluate, load_policy
    if args.command == "baseline":
        # baseline 命令使用相机估计和规则控制，不加载 PPO 模型。
        report = evaluate(episodes=args.episodes, seed=args.seed, randomize=args.randomize,
                          config_path=args.config, trace_path=Path(args.output)/"trace.csv")
        (Path(args.output)/"evaluation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["summary"], indent=2))
    elif args.command == "evaluate":
        # 评估同时比较规则基线与 PPO，并保存独立轨迹。
        policy = load_policy(args.model, config_path=args.config, allow_mismatch=args.allow_model_mismatch)
        report = compare(policy, episodes=args.episodes, seed=args.seed, config_path=args.config, output=args.output)
        if args.plot:
            from .plot import plot_evaluation
            print(f"Chart: {plot_evaluation(args.output, args.config)}")
        print(json.dumps({k: v["summary"] for k, v in report.items()}, indent=2))
    else:
        # view 是打开交互窗口的入口；训练和评估保持无窗口运行。
        from .env import PingPongEnv
        env = PingPongEnv(randomize=args.randomize, config_path=args.config)
        if args.sensor_fps is not None:
            # 该覆盖只对交互查看器生效。训练和评估继续使用配置帧率；查看器中
            # 昂贵的离屏双目渲染可降频，避免与被动 GUI 渲染线程争用资源。
            env.observer.settings["fps"] = args.sensor_fps
        policy = load_policy(args.model, config_path=args.config, allow_mismatch=args.allow_model_mismatch) if args.model else None
        obs, _ = env.reset(seed=args.seed)
        if args.snapshot:
            # 无窗口运行若干步后保存一帧真实仿真画面。
            for _ in range(24):
                action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            from PIL import Image
            path = Path(args.snapshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(path)
        else:
            from .viewer import camera_window, configure_camera
            # GUI 使用独立快照。物理、双相机渲染和 OpenCV 处理可能较慢，
            # 不让 viewer 直接占用正在更新的 env.data，拖动会更流畅。
            import mujoco
            display_data = mujoco.MjData(env.model)
            with camera_window(env.model, display_data) as viewer:
                with viewer.lock():
                    configure_camera(viewer.cam, env.model, env.config, args.camera)
                    mujoco.mj_copyData(display_data, env.model, env.data)
                if args.camera == "free":
                    print("Camera: free. Left drag: rotate; right drag: pan; wheel: zoom.")
                else:
                    print(f"Camera: fixed {args.camera}. Use --camera free for mouse navigation.")
                while viewer.is_running():
                    started = time.monotonic()
                    action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
                    # env.step 只更新物理数据；这里快速复制一份供 GUI 显示。
                    obs, _, done, truncated, info = env.step(action)
                    with viewer.lock():
                        mujoco.mj_copyData(display_data, env.model, env.data)
                    viewer.sync()
                    if done or truncated:
                        print(info)
                        with viewer.lock():
                            obs, _ = env.reset()
                            mujoco.mj_copyData(display_data, env.model, env.data)
                    time.sleep(max(0., 1/env.config["control"]["policy_hz"] - (time.monotonic()-started)))
        env.close()


if __name__ == "__main__":
    main()
