"""导出评估图表；训练 rollout 过程中不创建图形窗口。

绘图脚本只读取 `evaluation.json` 及对应 CSV 轨迹，并强制使用 Matplotlib
的非交互式 Agg 后端，因此可在服务器或 EGL 仿真环境中运行。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def plot_evaluation(directory, config_path=None):
    """根据一次评估目录生成基线与 PPO 的对比图。

    上排展示名义/随机化环境中的首个留出轨迹，下排展示成功率和顶点高度
    误差。函数返回生成的 PNG 路径，原始评估 JSON 和 CSV 文件不会被修改。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from rebotarm_description.parameters import description_path

    directory = Path(directory)
    # 评估结果和轨迹由 evaluation.py 写入，这里只做读取和绘图。
    report = json.loads((directory/"evaluation.json").read_text())
    config = yaml.safe_load(Path(config_path or description_path()/"config/pingpong.yaml").read_text())
    target = config["scene"]["home_surface_xyz_m"][2] + config["ball"]["radius_m"] + config["scene"]["target_apex_m"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ["#547b91", "#d9763d"]
    # 每个环境绘制一条 baseline 和一条 PPO 轨迹，便于观察控制误差。
    for mode, ax in zip(("nominal", "randomized"), axes[0]):
        for label, color in zip(("baseline", "ppo"), colors):
            with (directory/f"{mode}_{label}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            t = np.array([float(r["time_s"]) for r in rows])
            z = np.array([float(r["ball_z_m"]) for r in rows])
            ax.plot(t, z, label=label.upper(), color=color, linewidth=1.)
        ax.axhline(target, linestyle="--", linewidth=1., color="gray", label="target apex")
        ax.set(xlabel="Simulation time (s)", ylabel="Ball centre height (m)", title=f"{mode.capitalize()}: first held-out episode", xlim=(0, 5))
        ax.legend(fontsize=8)
    # 柱状图汇总两个环境下的成功率和平均顶点误差。
    x = np.arange(2)
    for label, offset, color in zip(("baseline", "ppo"), (-.18, .18), colors):
        values = [report[f"{mode}_{label}"]["summary"] for mode in ("nominal", "randomized")]
        axes[1, 0].bar(x+offset, [v["success_rate"]*100 for v in values], .36, label=label.upper(), color=color)
        axes[1, 1].bar(x+offset, [v["mean_apex_mae_m"]*1000 if v["mean_apex_mae_m"] is not None else np.nan for v in values], .36, label=label.upper(), color=color)
    axes[1, 0].set(ylabel="Success (%)", ylim=(0, 105), title="Survive the episode with at least 20 impacts")
    axes[1, 1].set(ylabel="Apex mean absolute error (mm)", title="Accuracy across held-out episodes")
    for ax in axes[1]:
        ax.set_xticks(x, ["Nominal", "Randomized"])
        ax.legend(fontsize=8)
    fig.savefig(directory/"comparison.png", dpi=160)
    plt.close(fig)
    return directory/"comparison.png"
