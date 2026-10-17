"""
最佳模型训练收敛曲线 — SCI 论文风格
=====================================
从 HPO history JSON 中提取最佳 trial 的 train/val loss 曲线绘制.
Y 轴使用对数坐标, 虚线样式, 与参考图风格一致.

输出:
  models/{model}/figures/training_curve.png/pdf  — 单模型图
  figures/training_curves_all.png/pdf            — 多模型合并图

Usage:
    python scripts/plot_training_curves.py                    # 全部模型
    python scripts/plot_training_curves.py --model ar fm_cfg  # 指定模型
"""

import os, sys, json, argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR    = os.path.join(ROOT_DIR, "figures")

# ═══════════════════════════════════════════════════════════════════
#  绘图配置
# ═══════════════════════════════════════════════════════════════════

FONT_SIZE_TITLE   = 22
FONT_SIZE_LABEL   = 20
FONT_SIZE_TICK    = 16
FONT_SIZE_LEGEND  = 16

TRAIN_COLOR       = "#2980B9"       # 蓝色 — Training
VAL_COLOR         = "#E67E22"       # 橙色 — Validation
TRAIN_STYLE       = "--"            # 虚线
VAL_STYLE         = "--"            # 虚线
LINE_WIDTH        = 1.5

SPINE_LW          = 1.5
TICK_WIDTH        = 1.5
TICK_LENGTH       = 5
GRID_ALPHA        = 0.3
GRID_LW           = 0.8

USE_LOG_Y         = True            # Y 轴对数坐标
SAVE_DPI          = 600
FIG_SIZE_SINGLE   = (8, 6)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "font.size":        FONT_SIZE_TICK,
    "mathtext.fontset": "stix",
    "axes.linewidth":   SPINE_LW,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "savefig.dpi":      SAVE_DPI,
    "savefig.bbox":     "tight",
})

# ═══════════════════════════════════════════════════════════════════
#  模型配置
# ═══════════════════════════════════════════════════════════════════

ALL_MODELS = ["fm_cfg", "ddpm", "ar", "cinn", "cvae", "cgan"]
DISPLAY = {
    "fm_cfg": "cFM-CFG", "ddpm": "DDPM", "ar": "cAR", "cinn": "cINN",
    "cvae": "cVAE", "cgan": "cWGAN-GP",
}


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════

def load_best_trial_history(model_name):
    """从 HPO history JSON 中提取最佳 trial 的训练曲线"""
    hist_path = os.path.join(MODELS_DIR, model_name, "results",
                              f"hpo_history_{model_name}.json")
    best_path = os.path.join(MODELS_DIR, model_name, "results",
                              f"hpo_best_{model_name}.json")

    if not os.path.exists(hist_path) or not os.path.exists(best_path):
        return None

    with open(hist_path) as f:
        all_trials = json.load(f)
    with open(best_path) as f:
        best_info = json.load(f)

    # 找最佳 trial (downstream MSE 最低)
    best_trial = min(all_trials, key=lambda t: t["value"])

    history = best_trial.get("history", {})
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        return None

    return {
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "trial_number": best_trial["number"],
        "downstream_mse": best_trial["value"],
        "stopped_epoch": best_trial.get("stopped_epoch", len(train_loss)),
    }


# ═══════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════

def plot_single(model_name, data):
    """单模型训练曲线"""
    fig_dir = os.path.join(MODELS_DIR, model_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    display = DISPLAY.get(model_name, model_name)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    # 检测是否有负值 (NLL 类模型: cINN, AR)
    has_neg = (train_loss < 0).any() or (val_loss < 0).any()

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    ax.plot(epochs, train_loss, TRAIN_STYLE, color=TRAIN_COLOR,
            lw=LINE_WIDTH, label="Training", zorder=2)
    ax.plot(epochs[:len(val_loss)], val_loss, VAL_STYLE, color=VAL_COLOR,
            lw=LINE_WIDTH, label="Validation", zorder=2)

    ax.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("NLL" if has_neg else "Loss", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=TICK_WIDTH, direction="in")

    # NLL 类模型有负值, 不使用对数坐标
    if USE_LOG_Y and not has_neg:
        ax.set_yscale("log")

    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9, edgecolor="#cccccc",
              loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"training_curve.{fmt}"), dpi=SAVE_DPI)
    plt.close(fig)
    print(f"    Saved training_curve.png/pdf -> {fig_dir}")


def plot_all(all_data):
    """多模型合并图 (子图网格)"""
    models = list(all_data.keys())
    n = len(models)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 5 * nrows),
                              constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        data = all_data[model_name]
        display = DISPLAY.get(model_name, model_name)

        train_loss = data["train_loss"]
        val_loss = data["val_loss"]
        has_neg = (train_loss < 0).any() or (val_loss < 0).any()
        epochs = np.arange(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, TRAIN_STYLE, color=TRAIN_COLOR,
                lw=LINE_WIDTH, label="Training")
        ax.plot(epochs[:len(val_loss)], val_loss, VAL_STYLE, color=VAL_COLOR,
                lw=LINE_WIDTH, label="Validation")

        ax.set_title(display, fontsize=FONT_SIZE_TITLE, fontweight="medium", pad=8)
        ax.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL - 4)
        ax.set_ylabel("NLL" if has_neg else "Loss", fontsize=FONT_SIZE_LABEL - 4)
        ax.tick_params(labelsize=FONT_SIZE_TICK - 3, width=1.0, direction="in")

        if USE_LOG_Y and not has_neg:
            ax.set_yscale("log")

        ax.legend(fontsize=FONT_SIZE_LEGEND - 4, framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    os.makedirs(FIG_DIR, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"training_curves_all.{fmt}"), dpi=SAVE_DPI)
    plt.close(fig)
    print(f"\n  Saved training_curves_all.png/pdf -> {FIG_DIR}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="最佳模型训练收敛曲线 (SCI 风格)")
    parser.add_argument("--model", nargs="*", default=None,
                        help="模型名称, 不指定则全部")
    args = parser.parse_args()

    models_to_run = args.model if args.model else ALL_MODELS

    print("=" * 60)
    print("  Training Convergence Curves (Best Trial)")
    print(f"  Models: {models_to_run}")
    print("=" * 60)

    all_data = {}
    for m in models_to_run:
        data = load_best_trial_history(m)
        if data is None:
            print(f"\n  [{m}] No history data, skip")
            continue

        display = DISPLAY.get(m, m)
        print(f"\n  [{display}] Trial #{data['trial_number']}, "
              f"MSE={data['downstream_mse']:.6f}, "
              f"Epochs={data['stopped_epoch']}")

        all_data[m] = data
        plot_single(m, data)

    if len(all_data) > 1:
        print("\n  Plotting combined figure...")
        plot_all(all_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
