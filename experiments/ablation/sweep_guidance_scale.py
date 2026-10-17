"""
Guidance Scale 敏感性扫参 — FM-CFG
====================================
固定 FM-CFG 训练权重 (fm_cfg_hpo.pth), 推理时扫不同 guidance_scale w,
评估用 ViT 性能预测器, 比较生成几何对应的预测性能与目标条件.

输出:
  experiments/ablation/results/sweep_guidance_scale.json
  experiments/ablation/figures/sweep_guidance_scale.png

Usage:
    python experiments/ablation/sweep_guidance_scale.py --gpu 0
"""

import os, sys, json, argparse
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# 与 scripts/plot_bubble.py 保持一致: K-NN 多样性 / 新颖性
K_NN = 5


def compute_diversity(gen_geoms, n_per_cond, k=K_NN):
    """对每组条件的生成样本计算 K-NN 多样性 (与 plot_bubble.py 一致)."""
    n_cond = len(gen_geoms) // n_per_cond
    div_list = []
    for i in range(n_cond):
        g = gen_geoms[i * n_per_cond:(i + 1) * n_per_cond]
        if len(g) > k:
            knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            knn.fit(g)
            d, _ = knn.kneighbors(g)
            div_list.append(float(d[:, 1:].mean()))
    return float(np.mean(div_list)) if div_list else 0.0


def compute_novelty(gen_geoms, train_geoms, k=K_NN):
    """生成样本到训练集的 K-NN 距离 (与 plot_bubble.py 一致)."""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(train_geoms)
    d, _ = knn.kneighbors(gen_geoms)
    return float(d.mean())

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用 plot_regression.py 的全局样式
from scripts.plot_regression import (
    SAVE_DPI, SAVE_FORMAT, SPINE_LW, TICK_DIRECTION,
    GRID_STYLE, GRID_LW, GRID_ALPHA, ANNO_BOX_EC,
    PERF_SYMBOL,
)
from shared.data_utils import get_dataloaders
from shared.evaluation import PerformanceEvaluator
from models.fm_cfg.model import ConditionalFlowMatching

RES_DIR = os.path.join(ROOT_DIR, "experiments", "ablation", "results")
FIG_DIR = os.path.join(ROOT_DIR, "experiments", "ablation", "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════
GUIDANCE_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

FM_CFG_CONFIG = {
    "hidden_dim": 512, "n_blocks": 5, "p_uncond": 0.061,
    "guidance_scale": 2.0, "sampling_steps": 100, "dropout": 0.081,
}
FM_CFG_CKPT = os.path.join(ROOT_DIR, "models", "fm_cfg", "checkpoints", "fm_cfg_hpo.pth")

N_COND, N_SAMPLES = 100, 10

# FM-CFG HPO 最佳值, 在图上用虚线标出
W_USED       = 2.0
W_USED_LABEL = r"cFM-CFG  $w=2.0$"

# ═══════════════════════════════════════════════════════════════════
#  绘图样式配置 ── 在此处修改字号 / 线宽 / 颜色等
# ═══════════════════════════════════════════════════════════════════

# 字号
FONT_LABEL          = 20
FONT_TICK           = 16
FONT_TITLE          = 22
FONT_LEGEND         = 16
FONT_MARK_TEXT      = 13

# 线条 / 标记
LINE_WIDTH          = 2.0
MARKER_SIZE         = 7

# FM-CFG 虚线标记
USED_LINE_COLOR     = "#5A5A5A"
USED_LINE_WIDTH     = 1.4
USED_LINE_ALPHA     = 0.75
USED_LABEL_BOX_FC   = "white"
USED_LABEL_BOX_EC   = "#999999"
USED_LABEL_BOX_LW   = 0.65
USED_LABEL_BOX_ALPHA = 0.94
USED_LABEL_Y_R2     = 0.35
USED_LABEL_Y_MSE    = 0.42    # 字号放大后, 0.55 会撞 title
USED_LABEL_Y_DIV    = 0.35

# 标题 pad
TITLE_PAD           = 10

# 图尺寸 + 边距
FIG_WIDTH           = 15
FIG_HEIGHT          = 4.8
MARGIN              = dict(left=0.07, right=0.98, top=0.90, bottom=0.15, wspace=0.30)

# legend 位置
LEGEND_LOC_R2       = "lower right"

# 配色
C_FLOW = "#2E5E8C"
C_EFF  = "#C0392B"
C_PR   = "#27AE60"
C_MSE  = "#8E44AD"
C_DIV  = "#16A085"

plt.rcParams.update({
    "font.family":   "serif",
    "font.serif":    ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": SPINE_LW,
    "xtick.direction": TICK_DIRECTION,
    "ytick.direction": TICK_DIRECTION,
    "xtick.top":     True,
    "ytick.right":   True,
    "savefig.dpi":   SAVE_DPI,
    "savefig.bbox":  "standard",   # 保持 figsize 一致, 不按内容裁剪
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--replot", action="store_true",
                        help="跳过 GPU 评估, 从已有 JSON 重绘图")
    args = parser.parse_args()

    if args.replot:
        json_path = os.path.join(RES_DIR, "sweep_guidance_scale.json")
        with open(json_path) as f:
            data = json.load(f)
        print(f"  [replot] Loading {json_path}")
        plot_sweep(data["results"])
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    bar = "=" * 60
    print(bar)
    print("  Guidance Scale Sweep — FM-CFG")
    print(f"  Scales: {GUIDANCE_SCALES}  |  Device: {device}")
    print(bar)

    model = ConditionalFlowMatching(geom_dim=21, cond_dim=3, **FM_CFG_CONFIG).to(device)
    if not os.path.exists(FM_CFG_CKPT):
        print(f"\n  [ERROR] Checkpoint not found: {FM_CFG_CKPT}")
        return
    model.load_state_dict(torch.load(FM_CFG_CKPT, map_location=device))
    model.eval()
    print(f"  Loaded FM-CFG ({sum(p.numel() for p in model.parameters()):,} params)")

    _, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
    stats = info["stats"]
    train_geoms = info["train_geom"]   # 用于 novelty 计算
    rng = np.random.RandomState(42)
    conditions = info["test_label"][rng.choice(len(info["test_label"]), N_COND, replace=False)]
    evaluator = PerformanceEvaluator(str(device))

    results = []
    for w in GUIDANCE_SCALES:
        def sample_fn(c, n, _m=model, _w=w):
            return _m.sample(c, n_samples=n, guidance_scale=_w)
        r = evaluator.evaluate_batch(sample_fn, conditions, stats, n_samples=N_SAMPLES)
        # K-NN 定义 (与 plot_bubble.py 一致)
        div_knn = compute_diversity(r["all_geom"], n_per_cond=N_SAMPLES)
        nov     = compute_novelty(r["all_geom"], train_geoms)
        entry = {
            "guidance_scale": w,
            "mse":        r["mse_mean"],
            "r2_flow":    r["r2_flow"],  "r2_eff":    r["r2_eff"],  "r2_pr":    r["r2_pr"],
            "nrmse_flow": r["nrmse_flow"],"nrmse_eff": r["nrmse_eff"],"nrmse_pr": r["nrmse_pr"],
            "diversity":  div_knn,
            "novelty":    nov,
        }
        results.append(entry)
        print(f"  w={w:4.1f}  MSE={r['mse_mean']:.6f}  "
              f"R2=[{r['r2_flow']:.4f},{r['r2_eff']:.4f},{r['r2_pr']:.4f}]  "
              f"Div={r['diversity_dist']:.4f}")

    out_json = os.path.join(RES_DIR, "sweep_guidance_scale.json")
    with open(out_json, "w") as f:
        json.dump({"model": "fm_cfg", "config": FM_CFG_CONFIG,
                   "n_cond": N_COND, "n_samples": N_SAMPLES,
                   "results": results}, f, indent=2)
    print(f"\n  Saved: {out_json}")

    plot_sweep(results)


def _mark_used(ax, x_used, label_text, label_y_axes=0.5):
    """灰虚线 + 沿线竖排标签."""
    ax.axvline(x_used, color=USED_LINE_COLOR, ls="--",
               lw=USED_LINE_WIDTH, alpha=USED_LINE_ALPHA, zorder=2.5)
    ax.text(x_used, label_y_axes, "  " + label_text,
            transform=ax.get_xaxis_transform(),
            rotation=90, ha="left", va="center", rotation_mode="anchor",
            fontsize=FONT_MARK_TEXT, color="#2C2C2C", family="serif",
            bbox=dict(boxstyle="round,pad=0.28",
                      fc=USED_LABEL_BOX_FC, ec=USED_LABEL_BOX_EC,
                      lw=USED_LABEL_BOX_LW, alpha=USED_LABEL_BOX_ALPHA),
            zorder=11)


def plot_sweep(results):
    scales = [r["guidance_scale"] for r in results]
    r2_flow = [r["r2_flow"] for r in results]
    r2_eff  = [r["r2_eff"]  for r in results]
    r2_pr   = [r["r2_pr"]   for r in results]
    mse     = [r["mse"]     for r in results]
    div     = [r["diversity"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(**MARGIN)

    # Panel 1: R²
    ax = axes[0]
    ax.plot(scales, r2_flow, "o-", color=C_FLOW, lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[0])
    ax.plot(scales, r2_eff,  "s-", color=C_EFF,  lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[1])
    ax.plot(scales, r2_pr,   "^-", color=C_PR,   lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[2])
    ax.set_xlabel(r"Guidance Scale $w$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$R^2$", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(prop={"size": FONT_LEGEND}, framealpha=0.94,
              edgecolor=ANNO_BOX_EC, loc=LEGEND_LOC_R2)
    ax.set_title("Regression Quality", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, W_USED, W_USED_LABEL, label_y_axes=USED_LABEL_Y_R2)

    # Panel 2: MSE_norm (log scale)
    ax = axes[1]
    ax.plot(scales, mse, "D-", color=C_MSE, lw=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel(r"Guidance Scale $w$", fontsize=FONT_LABEL)
    ax.set_ylabel("MSE (normalized)", fontsize=FONT_LABEL)
    ax.set_yscale("log")
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.set_title("Reconstruction Error", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, W_USED, W_USED_LABEL, label_y_axes=USED_LABEL_Y_MSE)

    # Panel 3: Diversity
    ax = axes[2]
    ax.plot(scales, div, "v-", color=C_DIV, lw=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel(r"Guidance Scale $w$", fontsize=FONT_LABEL)
    ax.set_ylabel("Sample Diversity", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.set_title("Sample Diversity", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, W_USED, W_USED_LABEL, label_y_axes=USED_LABEL_Y_DIV)

    out = os.path.join(FIG_DIR, f"sweep_guidance_scale.{SAVE_FORMAT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
