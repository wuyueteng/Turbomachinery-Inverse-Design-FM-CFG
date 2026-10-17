"""
p_uncond 训练扫参 — FM-CFG
============================
扫描训练阶段条件 dropout 概率 p_uncond ∈ {0.0, 0.05, 0.1, 0.2, 0.3},
每个值独立训练一个 FM-CFG 模型, 然后用 ViT 性能预测器评估.

机制说明:
  - p_uncond: 训练时随机将条件置零的概率, 直接控制 v_uncond 速度场的学习质量
  - 与 guidance_scale sweep 配对: p_uncond 是训练侧 CFG 超参, w 是推理侧

注意:
  - 训练 5 个模型, 单 GPU 约需 1-2 小时
  - 其他超参 (hidden_dim, n_blocks, lr, ...) 固定为 fm_cfg HPO 最佳值
  - 权重保存到 experiments/ablation/checkpoints/p_uncond_*.pth (不污染主目录)

输出:
  experiments/ablation/results/sweep_p_uncond.json
  experiments/ablation/figures/sweep_p_uncond.png
  experiments/ablation/checkpoints/p_uncond_*.pth

Usage:
    python experiments/ablation/sweep_p_uncond.py --gpu 0
    python experiments/ablation/sweep_p_uncond.py --gpu 0 --skip_train   # 跳过训练, 仅评估
"""

import os, sys, json, argparse, copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
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

from scripts.plot_regression import (
    SAVE_DPI, SAVE_FORMAT, SPINE_LW, TICK_DIRECTION,
    GRID_STYLE, GRID_LW, GRID_ALPHA, ANNO_BOX_EC,
    PERF_SYMBOL,
)
from shared.data_utils import get_dataloaders
from shared.evaluation import PerformanceEvaluator
from models.fm_cfg.model import ConditionalFlowMatching

RES_DIR  = os.path.join(ROOT_DIR, "experiments", "ablation", "results")
FIG_DIR  = os.path.join(ROOT_DIR, "experiments", "ablation", "figures")
CKPT_DIR = os.path.join(ROOT_DIR, "experiments", "ablation", "checkpoints")
for d in [RES_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════
P_UNCOND_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3]

# fm_cfg HPO 最佳超参 (排除 p_uncond, 由 sweep 控制)
BASE_CONFIG = {
    "hidden_dim":     512,
    "n_blocks":       5,
    "dropout":        0.081,
    "sampling_steps": 100,
    "guidance_scale": 2.0,   # 评估时使用主模型相同的 w (公平对比)
}
TRAIN_EPOCHS = 300
PATIENCE     = 60
LR           = 1e-4
BATCH_SIZE   = 256

N_COND, N_SAMPLES = 100, 10

# FM-CFG HPO 最佳值, 在图上用虚线标出 (HPO 找到 0.061; 与 sweep 网格点不完全重合)
P_USED       = 0.061
P_USED_LABEL = r"cFM-CFG  $p_{\mathrm{uncond}}=0.061$"

# ═══════════════════════════════════════════════════════════════════
#  绘图样式配置 ── 在此处修改字号 / 线宽 / 颜色等
# ═══════════════════════════════════════════════════════════════════

# 字号
FONT_LABEL          = 20    # X / Y 轴标题
FONT_TICK           = 16    # 主刻度
FONT_TITLE          = 22    # 子图顶部标题
FONT_LEGEND         = 16    # legend
FONT_MARK_TEXT      = 13    # FM-CFG 虚线沿线标签

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
USED_LABEL_Y_R2     = 0.25
USED_LABEL_Y_MSE    = 0.42    # 字号放大后, 0.55 会撞 title
USED_LABEL_Y_DIV    = 0.25

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
    "savefig.bbox":  "standard",   # 保持 figsize 一致, 不按内容裁剪 (避免不同图最终像素尺寸不同)
})


# ═══════════════════════════════════════════════════════════════════
#  训练单个 p_uncond
# ═══════════════════════════════════════════════════════════════════
def train_one(p_uncond, device, train_loader, val_loader):
    model = ConditionalFlowMatching(
        geom_dim=21, cond_dim=3,
        p_uncond=p_uncond, **BASE_CONFIG,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-6)

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        tl, n = 0.0, 0
        for batch in train_loader:
            geom  = batch["geometry"].to(device)
            label = batch["label"].to(device)
            loss = model.training_loss(geom, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * geom.shape[0]
            n  += geom.shape[0]
        train_loss = tl / n

        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for batch in val_loader:
                geom  = batch["geometry"].to(device)
                label = batch["label"].to(device)
                loss  = model.training_loss(geom, label)
                vl += loss.item() * geom.shape[0]
                vn += geom.shape[0]
        val_loss = vl / vn
        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 50 == 0:
            print(f"    epoch {epoch:>4d}  train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}")

        if no_improve >= PATIENCE and epoch > 100:
            print(f"    Early stop @ epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练, 加载已有 checkpoint 仅做评估")
    parser.add_argument("--replot", action="store_true",
                        help="跳过训练和评估, 从已有 JSON 重绘图")
    args = parser.parse_args()

    if args.replot:
        json_path = os.path.join(RES_DIR, "sweep_p_uncond.json")
        with open(json_path) as f:
            data = json.load(f)
        print(f"  [replot] Loading {json_path}")
        plot_sweep(data["results"])
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    bar = "=" * 60
    print(bar)
    print("  p_uncond Sweep — FM-CFG")
    print(f"  Values: {P_UNCOND_VALUES}  |  Device: {device}")
    print(f"  Base config: {BASE_CONFIG}")
    print(bar)

    train_loader, val_loader, _, info = get_dataloaders(batch_size=BATCH_SIZE, use_merged=True)
    stats = info["stats"]
    train_geoms = info["train_geom"]   # 用于 novelty 计算
    rng = np.random.RandomState(42)
    conditions = info["test_label"][rng.choice(len(info["test_label"]), N_COND, replace=False)]
    evaluator = PerformanceEvaluator(str(device))

    results = []
    for p in P_UNCOND_VALUES:
        print(f"\n  ── p_uncond = {p} ──")
        ckpt_path = os.path.join(CKPT_DIR, f"p_uncond_{p:.2f}.pth")

        if args.skip_train and os.path.exists(ckpt_path):
            print(f"    Loading existing {ckpt_path}")
            model = ConditionalFlowMatching(
                geom_dim=21, cond_dim=3, p_uncond=p, **BASE_CONFIG,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            best_val = None
        else:
            t0 = time.time()
            model, best_val = train_one(p, device, train_loader, val_loader)
            torch.save(model.state_dict(), ckpt_path)
            print(f"    Saved {ckpt_path}  ({time.time()-t0:.0f}s, best_val={best_val:.6f})")
        model.eval()

        def sample_fn(c, n, _m=model):
            return _m.sample(c, n_samples=n)

        r = evaluator.evaluate_batch(sample_fn, conditions, stats, n_samples=N_SAMPLES)
        # 用 plot_bubble.py 一致的 K-NN 定义覆盖 diversity, 并加入 novelty
        div_knn = compute_diversity(r["all_geom"], n_per_cond=N_SAMPLES)
        nov     = compute_novelty(r["all_geom"], train_geoms)
        entry = {
            "p_uncond": p,
            "best_val_loss": best_val,
            "mse":        r["mse_mean"],
            "r2_flow":    r["r2_flow"],  "r2_eff":    r["r2_eff"],  "r2_pr":    r["r2_pr"],
            "nrmse_flow": r["nrmse_flow"],"nrmse_eff": r["nrmse_eff"],"nrmse_pr": r["nrmse_pr"],
            "diversity":  div_knn,
            "novelty":    nov,
        }
        results.append(entry)
        print(f"    MSE={r['mse_mean']:.6f}  "
              f"R2=[{r['r2_flow']:.4f},{r['r2_eff']:.4f},{r['r2_pr']:.4f}]  "
              f"Div={r['diversity_dist']:.4f}")

        del model
        torch.cuda.empty_cache()

    out_json = os.path.join(RES_DIR, "sweep_p_uncond.json")
    with open(out_json, "w") as f:
        json.dump({"model": "fm_cfg", "base_config": BASE_CONFIG,
                   "train_epochs": TRAIN_EPOCHS,
                   "n_cond": N_COND, "n_samples": N_SAMPLES,
                   "results": results}, f, indent=2)
    print(f"\n  Saved: {out_json}")

    plot_sweep(results)


def _mark_used(ax, x_used, label_text, label_y_axes=0.5):
    """灰虚线 + 沿线竖排标签, 标出本模型实际采用的超参."""
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
    p_vals  = [r["p_uncond"]  for r in results]
    r2_flow = [r["r2_flow"]   for r in results]
    r2_eff  = [r["r2_eff"]    for r in results]
    r2_pr   = [r["r2_pr"]     for r in results]
    mse     = [r["mse"]       for r in results]
    div     = [r["diversity"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(**MARGIN)

    # Panel 1: R²
    ax = axes[0]
    ax.plot(p_vals, r2_flow, "o-", color=C_FLOW, lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[0])
    ax.plot(p_vals, r2_eff,  "s-", color=C_EFF,  lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[1])
    ax.plot(p_vals, r2_pr,   "^-", color=C_PR,   lw=LINE_WIDTH,
            markersize=MARKER_SIZE, label=PERF_SYMBOL[2])
    ax.set_xlabel(r"Conditional Dropout  $p_{\mathrm{uncond}}$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$R^2$", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(prop={"size": FONT_LEGEND}, framealpha=0.94,
              edgecolor=ANNO_BOX_EC, loc=LEGEND_LOC_R2)
    ax.set_title("Regression Quality", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, P_USED, P_USED_LABEL, label_y_axes=USED_LABEL_Y_R2)

    # Panel 2: MSE (log scale)
    ax = axes[1]
    ax.plot(p_vals, mse, "D-", color=C_MSE, lw=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel(r"Conditional Dropout  $p_{\mathrm{uncond}}$", fontsize=FONT_LABEL)
    ax.set_ylabel("MSE (normalized)", fontsize=FONT_LABEL)
    ax.set_yscale("log")
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.set_title("Reconstruction Error", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, P_USED, P_USED_LABEL, label_y_axes=USED_LABEL_Y_MSE)

    # Panel 3: Diversity
    ax = axes[2]
    ax.plot(p_vals, div, "v-", color=C_DIV, lw=LINE_WIDTH, markersize=MARKER_SIZE)
    ax.set_xlabel(r"Conditional Dropout  $p_{\mathrm{uncond}}$", fontsize=FONT_LABEL)
    ax.set_ylabel("Sample Diversity", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.set_title("Sample Diversity", fontsize=FONT_TITLE, fontweight="bold", pad=TITLE_PAD)
    _mark_used(ax, P_USED, P_USED_LABEL, label_y_axes=USED_LABEL_Y_DIV)

    out = os.path.join(FIG_DIR, f"sweep_p_uncond.{SAVE_FORMAT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
