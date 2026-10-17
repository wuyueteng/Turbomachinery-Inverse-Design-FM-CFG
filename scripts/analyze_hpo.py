"""
HPO 超参数重要性分析 + 优化历史绘图
=====================================
为每个模型生成:
  1. 超参数重要性条形图 (param_importance.png)
  2. 优化历史曲线 (optimization_history.png)
  3. 各 trial 训练曲线 (trial_training_curves.png)
  4. 超参数平行坐标图 (parallel_coordinate.png)
  5. 分析摘要 (hpo_analysis.txt)

所有结果保存至 models/{model}/figures/

Usage:
    python scripts/analyze_hpo.py                # 分析全部模型
    python scripts/analyze_hpo.py --model ar     # 分析单个模型
"""

import os, sys, json, argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.importance import get_param_importances

MODELS_DIR = os.path.join(ROOT_DIR, "models")
ALL_MODELS = ["ddpm_cfg", "ddpm", "fm_cfg", "cvae", "cgan", "ar", "cinn"]

DISPLAY = {
    "ddpm_cfg": "DDPM-CFG", "ddpm": "cDDPM", "fm_cfg": "cFM-CFG",
    "ar": "cAR", "cinn": "cINN", "cvae": "cVAE", "cgan": "cWGAN-GP",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "stix",
})


def analyze_one_model(model_name):
    """分析单个模型的 HPO 结果"""
    res_dir = os.path.join(MODELS_DIR, model_name, "results")
    fig_dir = os.path.join(MODELS_DIR, model_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    db_path = os.path.join(res_dir, f"optuna_hpo_{model_name}.db")
    hist_path = os.path.join(res_dir, f"hpo_history_{model_name}.json")
    best_path = os.path.join(res_dir, f"hpo_best_{model_name}.json")

    if not os.path.exists(db_path):
        print(f"  [{model_name}] No DB found, skip")
        return

    # Load study from DB
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=f"hpo_{model_name}", storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print(f"  [{model_name}] Only {len(completed)} completed trials, skip")
        return

    # Load history JSON
    with open(hist_path) as f:
        trial_history = json.load(f)
    with open(best_path) as f:
        best_info = json.load(f)

    display_name = DISPLAY.get(model_name, model_name)
    print(f"  [{display_name}] {len(completed)} completed, best MSE={study.best_value:.6f}")

    # ── 1. 超参数重要性 ──
    try:
        importances = get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, max(3, len(importances) * 0.6)))
        params = list(importances.keys())
        values = list(importances.values())
        colors = ["#E74C3C" if v == max(values) else "#3498DB" for v in values]
        bars = ax.barh(params, values, color=colors, edgecolor="black", lw=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlabel("Importance")
        ax.set_title(f"{display_name} — Hyperparameter Importance")
        ax.set_xlim(0, max(values) * 1.2)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "param_importance.png"), dpi=300)
        plt.close(fig)
        print(f"    Saved param_importance.png")
    except Exception as e:
        print(f"    Importance analysis failed: {e}")
        importances = {}

    # ── 2. 优化历史曲线 ──
    fig, ax = plt.subplots(figsize=(9, 5))
    trial_nums = [t.number for t in completed]
    trial_vals = [t.value for t in completed]
    best_so_far = []
    bsf = float("inf")
    for v in trial_vals:
        bsf = min(bsf, v)
        best_so_far.append(bsf)

    ax.scatter(trial_nums, trial_vals, c="#3498DB", s=40, alpha=0.6,
               edgecolors="black", lw=0.5, zorder=3, label="Trial MSE")
    ax.plot(trial_nums, best_so_far, c="#E74C3C", lw=2, zorder=4, label="Best so far")
    ax.axhline(study.best_value, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(max(trial_nums) * 0.95, study.best_value,
            f"Best: {study.best_value:.6f}", ha="right", va="bottom", fontsize=10,
            color="gray")

    # Mark pruned trials
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        ax.scatter([t.number for t in pruned],
                   [t.last_step if hasattr(t, 'last_step') else 0 for t in pruned],
                   c="lightgray", s=20, alpha=0.3, marker="x", label=f"Pruned ({len(pruned)})")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Downstream MSE")
    ax.set_title(f"{display_name} — Optimization History")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "optimization_history.png"), dpi=300)
    plt.close(fig)
    print(f"    Saved optimization_history.png")

    # ── 3. 各 trial 训练曲线 (top-5 + worst) ──
    sorted_trials = sorted(trial_history, key=lambda t: t["value"])
    top5 = sorted_trials[:5]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train loss
    ax = axes[0]
    for t in top5:
        if t.get("history") and t["history"].get("train_loss"):
            epochs = range(1, len(t["history"]["train_loss"]) + 1)
            ax.plot(epochs, t["history"]["train_loss"], lw=1.2, alpha=0.8,
                    label=f"T{t['number']} MSE={t['value']:.5f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title(f"{display_name} — Top-5 Trials (Train)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Val loss
    ax = axes[1]
    for t in top5:
        if t.get("history") and t["history"].get("val_loss"):
            epochs = range(1, len(t["history"]["val_loss"]) + 1)
            ax.plot(epochs, t["history"]["val_loss"], lw=1.2, alpha=0.8,
                    label=f"T{t['number']} MSE={t['value']:.5f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title(f"{display_name} — Top-5 Trials (Val)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "trial_training_curves.png"), dpi=300)
    plt.close(fig)
    print(f"    Saved trial_training_curves.png")

    # ── 4. 超参数平行坐标 (top-10 trials) ──
    top10 = sorted_trials[:min(10, len(sorted_trials))]
    if len(top10) >= 3:
        param_names = list(top10[0]["params"].keys())
        fig, ax = plt.subplots(figsize=(max(8, len(param_names) * 1.5), 5))

        # Normalize each param to [0,1] for visualization
        param_vals = {p: [] for p in param_names}
        for t in top10:
            for p in param_names:
                param_vals[p].append(t["params"].get(p, 0))

        param_min = {p: min(v) for p, v in param_vals.items()}
        param_max = {p: max(v) for p, v in param_vals.items()}

        for idx, t in enumerate(top10):
            y_norm = []
            for p in param_names:
                pmin, pmax = param_min[p], param_max[p]
                rng = pmax - pmin if pmax > pmin else 1
                y_norm.append((t["params"].get(p, 0) - pmin) / rng)
            color = plt.cm.RdYlGn_r(idx / max(len(top10) - 1, 1))
            ax.plot(range(len(param_names)), y_norm, "o-", color=color, lw=1.5,
                    alpha=0.7, markersize=5, label=f"T{t['number']} ({t['value']:.5f})")

        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=30, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title(f"{display_name} — Top-10 Parallel Coordinates")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "parallel_coordinate.png"), dpi=300)
        plt.close(fig)
        print(f"    Saved parallel_coordinate.png")

    # ── 5. 分析摘要 ──
    lines = [
        f"HPO Analysis: {display_name}",
        "=" * 50,
        f"Total trials: {len(study.trials)}",
        f"Completed: {len(completed)}",
        f"Pruned: {len(pruned)}",
        f"Best MSE: {study.best_value:.6f} (Trial #{study.best_trial.number})",
        f"Best params: {json.dumps(study.best_params, indent=2)}",
        "",
        "Hyperparameter Importance:",
    ]
    for p, v in importances.items():
        lines.append(f"  {p:<20s}: {v:.4f}")
    lines.extend([
        "",
        "Top-5 Trials:",
        f"{'Trial':>6s}  {'MSE':>12s}  {'Epoch':>6s}  {'Params':>10s}",
        "-" * 40,
    ])
    for t in sorted_trials[:5]:
        lines.append(f"  #{t['number']:<4d}  {t['value']:>12.6f}  "
                     f"{str(t.get('stopped_epoch', '?')):>6s}  "
                     f"{t.get('n_params', 0):>10,}")

    with open(os.path.join(fig_dir, "hpo_analysis.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"    Saved hpo_analysis.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["all"] + ALL_MODELS)
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    print("=" * 50)
    print("  HPO Analysis & Visualization")
    print("=" * 50)

    for m in models:
        print(f"\n--- {DISPLAY.get(m, m)} ---")
        analyze_one_model(m)

    print("\nDone!")


if __name__ == "__main__":
    main()
