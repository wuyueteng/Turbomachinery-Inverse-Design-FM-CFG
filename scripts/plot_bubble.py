"""
Bubble Chart: Diversity vs MRE, bubble size = Novelty
=======================================================
三维气泡图, 比较所有模型:
  - X 轴: Diversity (样本间 K-NN 距离)
  - Y 轴: MRE (平均相对误差, %)
  - 气泡大小: Novelty (与训练集的距离)

直接读取 inference.py 的输出计算指标 (无需 GPU).

Output: figures/bubble_chart.png

Usage:
    python scripts/plot_bubble.py                           # 从 inference 输出计算 + 绘图
    python scripts/plot_bubble.py --model ar fm_cfg  # 指定模型
    python scripts/plot_bubble.py --cache                   # 直接用缓存 json 绘图
"""

import os, sys, json, argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from shared.data_utils import get_dataloaders
from sklearn.neighbors import NearestNeighbors

# ═══════════════════════════════════════════════════════════════════
#  全局绘图配置
# ═══════════════════════════════════════════════════════════════════

# ── 字体 ──
FONT_FAMILY       = "serif"
FONT_SERIF        = ["Times New Roman"]
MATHTEXT_FONTSET  = "stix"

# ── 字号 ──
FONT_SIZE_TITLE   = 20              # 图标题
FONT_SIZE_LABEL   = 22              # 坐标轴标签
FONT_SIZE_TICK    = 18              # 刻度数字
FONT_SIZE_ARROW   = 18              # better-direction 提示
FONT_SIZE_LEGEND  = 16              # 图例字号

# ── 气泡 ──
BUBBLE_SIZE_MIN   = 220             # 最小气泡 (Novelty 最低)
BUBBLE_SIZE_MAX   = 1100            # 最大气泡 (Novelty 最高)
BUBBLE_ALPHA      = 0.78            # 气泡填充透明度
BUBBLE_EDGE_COLOR = "#1f2933"       # 气泡边框颜色
BUBBLE_EDGE_WIDTH = 1.3             # 气泡边框线宽
BUBBLE_SHADOW     = True            # 是否给气泡添加柔和阴影

# ── 网格 ──
GRID_ALPHA        = 0.22
GRID_STYLE        = "--"
GRID_LW           = 0.55
MINOR_GRID_ALPHA  = 0.10

# ── 刻度 & 边框 ──
TICK_DIRECTION    = "in"
TICK_LENGTH       = 5
TICK_WIDTH        = 0.8
SPINE_LW          = 1.0

# ── "better" 区域阴影 ──
IDEAL_REGION_COLOR = "#2ECC71"      # 理想区 (高 Diversity + 低 MRE) 阴影色
IDEAL_REGION_ALPHA = 0.06           # 阴影透明度
IDEAL_ARROW_COLOR  = "#1f2933"      # better 方向箭头颜色

# ── 输出 ──
SAVE_DPI          = 600
SAVE_FORMAT       = "png"           # "png" / "pdf" / "svg"
FIG_SIZE           = (8.5, 7)

# ── 应用 rcParams ──
plt.rcParams.update({
    "font.family":      FONT_FAMILY,
    "font.serif":       FONT_SERIF,
    "mathtext.fontset": MATHTEXT_FONTSET,
    "axes.linewidth":   SPINE_LW,
    "xtick.direction":  TICK_DIRECTION,
    "ytick.direction":  TICK_DIRECTION,
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

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALL_MODELS = ["pa_fm_cfg", "fm_cfg", "fm", "ddpm", "ar", "cinn", "cvae", "cgan", "ddpm_cfg"]

# ── 默认绘制哪些模型 (命令行 --model 优先, 不传则用此列表) ──
# 注释掉不需要的模型即可
PLOT_MODELS = [
    # "pa_fm_cfg",
    "fm_cfg",
    # "fm",
    "ddpm",
    # "ddpm_cfg",
    "ar",
    "cinn",
    "cvae",
    "cgan",
]
DISPLAY = {
    "pa_fm_cfg": "PA-cFM-CFG", "fm_cfg": "cFM-CFG", "fm": "FM",
    "ddpm": "cDDPM", "ddpm_cfg": "DDPM-CFG", "ar": "cAR", "cinn": "cINN",
    "cvae": "cVAE", "cgan": "cWGAN-GP",
}
# ── 图例显示顺序 (与 ALL_MODELS / PLOT_MODELS 解耦) ──
LEGEND_ORDER = ["fm_cfg", "ar", "cvae", "cgan", "ddpm", "cinn"]
COLORS = {
    "pa_fm_cfg": "#C0392B", "fm_cfg": "#8E44AD", "fm": "#E67E22",
    "ddpm": "#2C3E50", "ddpm_cfg": "#27AE60", "ar": "#2980B9", "cinn": "#7F8C8D",
    "cvae": "#16A085", "cgan": "#D35400",
}
K_NN = 5


# ═══════════════════════════════════════════════════════════════════
#  从 inference.py 输出计算气泡图指标 (无需 GPU)
# ═══════════════════════════════════════════════════════════════════

def _compute_diversity(geom, n_per_cond, k=K_NN):
    """对每组条件的生成样本计算 K-NN 多样性"""
    n_cond = len(geom) // n_per_cond
    div_list = []
    for i in range(n_cond):
        g = geom[i * n_per_cond:(i + 1) * n_per_cond]
        if len(g) > k:
            knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            knn.fit(g)
            d, _ = knn.kneighbors(g)
            div_list.append(float(d[:, 1:].mean()))
    return float(np.mean(div_list)) if div_list else 0.0


def _compute_novelty(geom, train_geom, k=K_NN):
    """生成样本到训练集的 K-NN 距离 (novelty)"""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(train_geom)
    d, _ = knn.kneighbors(geom)
    return float(d.mean())


def load_from_inference(models_to_run=None, n_per_cond=10):
    """
    从 inference.py 的输出直接计算气泡图所需指标 (无需 GPU).
    读取: models/{model}/results/inference/ 下的 npy + json.
    """
    models_to_run = models_to_run or ALL_MODELS

    # 加载训练集几何 (用于 novelty 计算)
    _, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
    train_geom = info["train_geom"]

    results = {}
    for m in models_to_run:
        inf_dir = os.path.join(MODELS_DIR, m, "results", "inference")
        geom_path = os.path.join(inf_dir, "geometry_norm.npy")
        pred_path = os.path.join(inf_dir, "pred_performance.npy")
        targ_path = os.path.join(inf_dir, "target_performance.npy")
        json_path = os.path.join(inf_dir, "inference_metrics.json")

        if not all(os.path.exists(p) for p in [geom_path, pred_path, targ_path]):
            print(f"  [{m}] No inference output, skip (run inference.py --model {m} first)")
            continue

        print(f"  [{DISPLAY.get(m, m)}] loading inference data...")
        geom = np.load(geom_path)
        pred = np.load(pred_path)
        targ = np.load(targ_path)

        # MRE (从 json 读取，或重新计算)
        if os.path.exists(json_path):
            with open(json_path) as f:
                met = json.load(f)
            mre = (met.get("mre_flow", 0) + met.get("mre_eff", 0) + met.get("mre_pr", 0)) / 3
        else:
            rel_err = np.abs(pred - targ) / (np.abs(targ) + 1e-12)
            mre = float(np.mean(rel_err) * 100)

        # Diversity & Novelty
        diversity = _compute_diversity(geom, n_per_cond)
        novelty = _compute_novelty(geom, train_geom)

        results[m] = {
            "mre": mre,
            "diversity_mean": diversity,
            "novelty_mean": novelty,
        }
        print(f"    MRE={mre:.2f}%  Div={diversity:.4f}  Nov={novelty:.4f}")

    # 缓存
    if results:
        cache_path = os.path.join(FIG_DIR, "eval_bubble_metrics.json")
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {cache_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  气泡图绘制
# ═══════════════════════════════════════════════════════════════════

def plot_bubble(metrics):
    """Diversity vs MRE, bubble size = Novelty.

    指标方向:
      X 轴 Diversity ↑ better
      Y 轴 MRE       ↓ better
      Bubble size = Novelty ↑ better
    所以理想模型 = 右下角 + 大气泡.
    """
    models = [m for m in ALL_MODELS if m in metrics]
    if not models:
        print("  No model data available.")
        return

    # ── 收集数据 ──
    divs  = np.array([metrics[m]["diversity_mean"] for m in models])
    mres  = np.array([metrics[m]["mre"] for m in models])
    novs  = np.array([metrics[m]["novelty_mean"] for m in models])

    # Novelty -> bubble size (线性映射)
    if novs.max() > novs.min():
        nov_norm = (novs - novs.min()) / (novs.max() - novs.min())
    else:
        nov_norm = np.full_like(novs, 0.5)
    sizes = BUBBLE_SIZE_MIN + nov_norm * (BUBBLE_SIZE_MAX - BUBBLE_SIZE_MIN)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFBFD")

    # ── 数据范围 → 留出图例与箭头空间的坐标轴 ──
    x_pad = (divs.max() - divs.min()) * 0.18 if divs.max() > divs.min() else 0.05
    y_pad = (mres.max() - mres.min()) * 0.22 if mres.max() > mres.min() else 0.5
    x_lo, x_hi = divs.min() - x_pad * 0.6, divs.max() + x_pad * 1.4
    y_lo, y_hi = mres.min() - y_pad * 0.6, mres.max() + y_pad * 1.2
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # ── 理想区域阴影 (右下角: 高 Diversity + 低 MRE) ──
    x_mid = divs.mean()
    y_mid = mres.mean()
    ideal = Rectangle((x_mid, y_lo), x_hi - x_mid, y_mid - y_lo,
                      facecolor=IDEAL_REGION_COLOR, alpha=IDEAL_REGION_ALPHA,
                      edgecolor="none", zorder=0)
    ax.add_patch(ideal)
    ax.text(x_hi - (x_hi - x_mid) * 0.04, y_lo + (y_mid - y_lo) * 0.06,
            "Ideal region", ha="right", va="bottom",
            fontsize=FONT_SIZE_ARROW - 1, color="#1E8449",
            fontstyle="italic", alpha=0.85, zorder=1)

    # ── 气泡 (带柔和阴影) ──
    for i, mk in enumerate(models):
        if BUBBLE_SHADOW:
            ax.scatter(divs[i], mres[i], s=sizes[i] * 1.05,
                       color="#000000", alpha=0.12,
                       edgecolors="none", zorder=4)
        ax.scatter(divs[i], mres[i], s=sizes[i],
                   color=COLORS[mk], alpha=BUBBLE_ALPHA,
                   edgecolors=BUBBLE_EDGE_COLOR, linewidth=BUBBLE_EDGE_WIDTH,
                   zorder=5)

    # ── 坐标轴 ──
    ax.set_xlabel(r"Diversity  $D_{div}^{k}$  ($\uparrow$ better)",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"MRE [%]  ($\downarrow$ better)",
                  fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=2.5, width=0.6, direction=TICK_DIRECTION)

    # ── 网格 ──
    ax.grid(True, which="major", alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_LW)
    ax.grid(True, which="minor", alpha=MINOR_GRID_ALPHA, linestyle=":", linewidth=0.4)
    ax.set_axisbelow(True)

    # ── 边框 ──
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_color("#333333")

    # ── 图例 1: 模型颜色 (放在图内左上空白区) ──
    legend_models = [m for m in LEGEND_ORDER if m in metrics]
    # 兜底: LEGEND_ORDER 未覆盖的模型按 ALL_MODELS 顺序追加
    legend_models += [m for m in models if m not in legend_models]
    model_handles = [
        Line2D([], [], marker="o", linestyle="",
               markersize=10, markerfacecolor=COLORS[mk],
               markeredgecolor=BUBBLE_EDGE_COLOR, markeredgewidth=1.0,
               alpha=BUBBLE_ALPHA, label=DISPLAY[mk])
        for mk in legend_models
    ]
    leg_models = ax.legend(handles=model_handles,
                           title="Model",
                           title_fontsize=FONT_SIZE_LEGEND,
                           prop={"size": FONT_SIZE_LEGEND - 1},
                           loc="upper left", bbox_to_anchor=(0.015, 0.985),
                           framealpha=0.92, edgecolor="#cccccc",
                           borderpad=0.8, labelspacing=0.7,
                           handletextpad=0.6)
    leg_models.get_frame().set_linewidth(0.8)
    ax.add_artist(leg_models)

    # ── 图例 2: 气泡大小 = Novelty (放在图内左上, 模型图例右侧) ──
    legend_novs = [novs.min(), np.median(novs), novs.max()]
    legend_labels = [f"{v:.3f}" for v in legend_novs]
    if novs.max() > novs.min():
        legend_sizes = [BUBBLE_SIZE_MIN + (v - novs.min()) / (novs.max() - novs.min())
                        * (BUBBLE_SIZE_MAX - BUBBLE_SIZE_MIN) for v in legend_novs]
    else:
        legend_sizes = [BUBBLE_SIZE_MIN, (BUBBLE_SIZE_MIN + BUBBLE_SIZE_MAX) / 2, BUBBLE_SIZE_MAX]

    size_handles = [
        Line2D([], [], marker="o", linestyle="",
               markersize=np.sqrt(s) * 0.55,
               markerfacecolor="#B0B7BF",
               markeredgecolor=BUBBLE_EDGE_COLOR, markeredgewidth=1.0,
               alpha=BUBBLE_ALPHA)
        for s in legend_sizes
    ]
    leg_size = ax.legend(size_handles, legend_labels,
                         title=r"Novelty  $D_{nov}^{k}$",
                         title_fontsize=FONT_SIZE_LEGEND,
                         prop={"size": FONT_SIZE_LEGEND - 1},
                         loc="upper right", bbox_to_anchor=(0.54, 0.985),
                         framealpha=0.92, edgecolor="#cccccc",
                         labelspacing=1.8, borderpad=0.9,
                         handletextpad=1.3)
    leg_size.get_frame().set_linewidth(0.8)

    # ── 保存 ──
    out_path = os.path.join(FIG_DIR, f"bubble_chart.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bubble Chart: Diversity vs MRE vs Novelty")
    parser.add_argument("--model", nargs="*", default=None,
                        help="模型名称 (可多个), 不指定则使用代码中 PLOT_MODELS 列表")
    parser.add_argument("--cache", action="store_true",
                        help="直接读取缓存 json, 跳过重新计算")
    args = parser.parse_args()

    # 优先级: 命令行 --model > 代码顶部 PLOT_MODELS
    models_to_plot = args.model if args.model else PLOT_MODELS

    cache_path = os.path.join(FIG_DIR, "eval_bubble_metrics.json")

    if args.cache and os.path.exists(cache_path):
        print("Loading cached metrics...")
        with open(cache_path) as f:
            metrics = json.load(f)
    else:
        print("Computing from inference output...")
        metrics = load_from_inference(models_to_run=models_to_plot)

    # 只保留要绘制的模型
    metrics = {m: v for m, v in metrics.items() if m in models_to_plot}

    print("\nPlotting bubble chart...")
    plot_bubble(metrics)

    # Summary table
    print("\n" + "=" * 80)
    print(f"  {'Model':<14s} {'MRE%':>8s} {'Diversity':>10s} {'Novelty':>10s}")
    print("  " + "-" * 74)
    for m in ALL_MODELS:
        if m not in metrics:
            continue
        mt = metrics[m]
        print(f"  {DISPLAY[m]:<14s} {mt['mre']:>7.2f}% "
              f"{mt['diversity_mean']:>10.4f} {mt['novelty_mean']:>10.4f}")
    print("  " + "=" * 74)
    print("\nDone!")


if __name__ == "__main__":
    main()
