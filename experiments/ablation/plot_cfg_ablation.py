"""
CFG 消融对比图 — FM vs FM-CFG
==============================
1×3 分组柱状图: R² | nRMSE | Diversity + Novelty
配合分别保存的 3 个独立子图 (Word 手动排版用).

多样性与新颖性按 scripts/plot_bubble.py 一致的 K-NN 定义计算 (K=5).

输出:
  experiments/ablation/figures/cfg_ablation_bar.png      (1×3 合并)
  experiments/ablation/figures/panels/panel_cfg_r2.png
  experiments/ablation/figures/panels/panel_cfg_nrmse.png
  experiments/ablation/figures/panels/panel_cfg_divnov.png

Usage:
    python experiments/ablation/plot_cfg_ablation.py
"""

import os, sys, json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用 plot_regression.py 的 rcParams 全局样式 (Times/stix/DPI 等)
from scripts.plot_regression import (
    FIG_WIDTH_SUMMARY, SAVE_DPI, SAVE_FORMAT,
    ANNO_BOX_EC, GRID_STYLE, GRID_LW, GRID_ALPHA,
    PERF_SYMBOL,
)
# K-NN 多样性 / 新颖性 (与 plot_bubble.py 完全一致的定义)
from scripts.plot_bubble import _compute_diversity, _compute_novelty, K_NN
from shared.data_utils import get_dataloaders

# 本地覆盖: 保持 figsize 严格一致 (不按内容裁剪, 避免不同图 PNG 像素尺寸不同)
plt.rcParams["savefig.bbox"] = "standard"


# ═══════════════════════════════════════════════════════════════════
#  样式配置 ── 改这里即可统一调整全图风格
# ═══════════════════════════════════════════════════════════════════

# ── 字号: combined 1×3 图 (panel 较紧凑, 字号小一档) ──
COMBINED_TITLE_FONT     = 18    # 顶部子图标题 (R² / nRMSE / Diversity & Novelty)
COMBINED_LEGEND_FONT    = 13    # legend (FM / FM-CFG)
COMBINED_TICK_FONT      = 12    # Y 轴刻度数字
COMBINED_ANNO_FONT      = 11    # 柱顶数值
COMBINED_XSYMBOL_FONT   = 17    # X 轴数学符号 ṁ, η, π
COMBINED_XTEXT_FONT     = 13    # X 轴普通文字 "Diversity", "Novelty"

# ── 字号: individual 单独子图 (大尺寸输出, 字号大一档) ──
PANEL_YLABEL_FONT       = 24    # Y 轴指标名 (替代顶部标题)
PANEL_LEGEND_FONT       = 17
PANEL_TICK_FONT         = 18
PANEL_ANNO_FONT         = 16
PANEL_XSYMBOL_FONT      = 24    # ṁ, η, π (与 Y 轴 label 字号保持一致)
PANEL_XTEXT_FONT        = 24    # "Diversity", "Novelty" (与 Y 轴 label 字号保持一致)

# ── 柱样式 ──
BAR_WIDTH               = 0.34
BAR_ALPHA               = 0.85
BAR_EDGE_COLOR          = "black"
BAR_EDGE_WIDTH          = 0.6

# ── 顶部 Y 轴余量 (柱顶数字 + legend 不会被截断) ──
YLIM_PAD_R2_NRMSE       = 0.12   # R² / nRMSE 面板
YLIM_PAD_DIVNOV         = 0.18   # divnov 面板 (legend 在 upper right, 需更多余量)

# ── 标题 pad ──
TITLE_PAD               = 10

# ── 图尺寸 (inches) ──
COMBINED_FIG_WIDTH      = FIG_WIDTH_SUMMARY    # 15
COMBINED_FIG_HEIGHT     = 4.8
PANEL_FIG_WIDTH         = 6.5     # 所有 panel 统一宽度 (cfg 柱状图 divnov 是单 Y 轴, 无需加宽)
PANEL_FIG_HEIGHT        = 5.5     # 所有 panel 等高

# ── subplot 边距 ──
COMBINED_MARGIN         = dict(left=0.06, right=0.98, top=0.90, bottom=0.13, wspace=0.24)
PANEL_MARGIN            = dict(left=0.18, right=0.96, top=0.96, bottom=0.13)

# ── legend 位置 (kind 决定, 避免遮挡数据) ──
LEGEND_LOC = {"r2": "lower left", "nrmse": "upper left", "divnov": "upper right"}

# ── 多样性 / 新颖性数学符号 (与 scripts/plot_bubble.py 一致) ──
DIV_SYMBOL = r"$D_{div}^{k}$"     # Diversity: 条件内 K-NN 距离
NOV_SYMBOL = r"$D_{nov}^{k}$"     # Novelty:   到训练集 K-NN 距离

# ── 各 panel 的标题 / Y 轴标签文字 (combined 用 title, individual 用 ylabel) ──
PANEL_TITLES = {
    "r2":     r"$R^2$",
    "nrmse":  "nRMSE",
    "divnov": "Diversity & Novelty",   # combined 顶部标题
}
PANEL_YLABELS = {
    "r2":     r"$R^2$",
    "nrmse":  "nRMSE",
    "divnov": "K-NN Distance",          # individual 子图 Y 轴
}

# ── divnov 面板 X 轴刻度: 单行 "名称 符号" ──
DIVNOV_XTICK_LABELS = [
    f"Diversity  {DIV_SYMBOL}",
    f"Novelty  {NOV_SYMBOL}",
]

# ── 模型对 + 配色 ──
PAIR = [("fm", "FM"), ("fm_cfg", "cFM-CFG")]
COLORS = {"fm": "#E67E22", "fm_cfg": "#8E44AD"}

# ── 性能维度顺序: 流量 → 压比 → 效率 ──
PERF_ORDER_KEYS    = ["flow", "pr", "eff"]
PERF_ORDER_SYMBOLS = [PERF_SYMBOL[0], PERF_SYMBOL[2], PERF_SYMBOL[1]]

# ── 评估 ──
N_PER_COND = 10

# ── 路径 ──
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR    = os.path.join(ROOT_DIR, "experiments", "ablation", "figures")
PANEL_DIR  = os.path.join(FIG_DIR, "panels")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════
def load_metrics(model_name):
    path = os.path.join(MODELS_DIR, model_name, "results", "inference", "inference_metrics.json")
    with open(path) as f:
        return json.load(f)


def load_geom(model_name):
    return np.load(os.path.join(MODELS_DIR, model_name, "results", "inference", "geometry_norm.npy"))


def compute_div_nov(model_name, train_geom):
    """K-NN 多样性 + 新颖性, K=5."""
    geom = load_geom(model_name)
    div = _compute_diversity(geom, n_per_cond=N_PER_COND, k=K_NN)
    nov = _compute_novelty(geom, train_geom, k=K_NN)
    return div, nov


# ═══════════════════════════════════════════════════════════════════
#  单个面板绘制
# ═══════════════════════════════════════════════════════════════════
def draw_bar_panel(ax, kind, data, mode, show_legend=False):
    """
    kind ∈ {'r2', 'nrmse', 'divnov'}
    mode ∈ {'combined', 'panel'}
    """
    # 根据 mode 选择字号集
    if mode == "combined":
        fs_title   = COMBINED_TITLE_FONT
        fs_legend  = COMBINED_LEGEND_FONT
        fs_tick    = COMBINED_TICK_FONT
        fs_anno    = COMBINED_ANNO_FONT
        fs_xsymbol = COMBINED_XSYMBOL_FONT
        fs_xtext   = COMBINED_XTEXT_FONT
        fs_ylabel  = None
        use_title  = True
    else:  # "panel"
        fs_title   = None
        fs_legend  = PANEL_LEGEND_FONT
        fs_tick    = PANEL_TICK_FONT
        fs_anno    = PANEL_ANNO_FONT
        fs_xsymbol = PANEL_XSYMBOL_FONT
        fs_xtext   = PANEL_XTEXT_FONT
        fs_ylabel  = PANEL_YLABEL_FONT
        use_title  = False

    # 按 kind 准备数据
    if kind == "r2":
        keys       = [f"r2_{k}"    for k in PERF_ORDER_KEYS]
        sub_labels = PERF_ORDER_SYMBOLS
        fs_xt      = fs_xsymbol
        v_no       = [data["fm"][k]     for k in keys]
        v_yes      = [data["fm_cfg"][k] for k in keys]
    elif kind == "nrmse":
        keys       = [f"nrmse_{k}" for k in PERF_ORDER_KEYS]
        sub_labels = PERF_ORDER_SYMBOLS
        fs_xt      = fs_xsymbol
        v_no       = [data["fm"][k]     for k in keys]
        v_yes      = [data["fm_cfg"][k] for k in keys]
    elif kind == "divnov":
        sub_labels = DIVNOV_XTICK_LABELS
        fs_xt      = fs_xtext
        v_no       = [data["fm"]["div"],     data["fm"]["nov"]]
        v_yes      = [data["fm_cfg"]["div"], data["fm_cfg"]["nov"]]
    else:
        raise ValueError(kind)

    n = len(sub_labels)
    x = np.arange(n)

    bars_no = ax.bar(x - BAR_WIDTH / 2, v_no, BAR_WIDTH, label=PAIR[0][1],
                     color=COLORS[PAIR[0][0]], alpha=BAR_ALPHA,
                     edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    bars_yes = ax.bar(x + BAR_WIDTH / 2, v_yes, BAR_WIDTH, label=PAIR[1][1],
                      color=COLORS[PAIR[1][0]], alpha=BAR_ALPHA,
                      edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)

    # 柱顶数值
    for bar in list(bars_no) + list(bars_yes):
        h = bar.get_height()
        fmt = f"{h:.3f}" if abs(h) >= 0.1 else f"{h:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h, fmt,
                ha="center", va="bottom", fontsize=fs_anno)

    # 顶部 Y 轴余量
    ymin, ymax = ax.get_ylim()
    pad = YLIM_PAD_DIVNOV if kind == "divnov" else YLIM_PAD_R2_NRMSE
    ax.set_ylim(ymin, ymax + (ymax - ymin) * pad)

    # X 轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=fs_xt)

    # 顶部标题 (combined) 或 Y 轴标签 (panel)
    if use_title:
        ax.set_title(PANEL_TITLES[kind], fontsize=fs_title, pad=TITLE_PAD)
    else:
        ax.set_ylabel(PANEL_YLABELS[kind], fontsize=fs_ylabel)

    # 网格 + Y 轴刻度数字 (X 轴刻度已通过 set_xticklabels 单独设置, 这里只调 Y)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.grid(True, axis="y", linestyle=GRID_STYLE, linewidth=GRID_LW,
            alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)

    # legend
    if show_legend:
        ax.legend(prop={"size": fs_legend},
                  loc=LEGEND_LOC.get(kind, "best"),
                  framealpha=0.94, edgecolor=ANNO_BOX_EC)


# ═══════════════════════════════════════════════════════════════════
#  整张 1×3 图
# ═══════════════════════════════════════════════════════════════════
def make_combined(data):
    fig, axes = plt.subplots(1, 3, figsize=(COMBINED_FIG_WIDTH, COMBINED_FIG_HEIGHT))
    fig.subplots_adjust(**COMBINED_MARGIN)

    draw_bar_panel(axes[0], "r2",     data, "combined", show_legend=True)
    draw_bar_panel(axes[1], "nrmse",  data, "combined")
    draw_bar_panel(axes[2], "divnov", data, "combined", show_legend=True)

    out = os.path.join(FIG_DIR, f"cfg_ablation_bar.{SAVE_FORMAT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved combined : {out}")


# ═══════════════════════════════════════════════════════════════════
#  3 个独立子图
# ═══════════════════════════════════════════════════════════════════
def make_panels(data):
    for kind in ["r2", "nrmse", "divnov"]:
        fig, ax = plt.subplots(figsize=(PANEL_FIG_WIDTH, PANEL_FIG_HEIGHT))
        fig.subplots_adjust(**PANEL_MARGIN)
        draw_bar_panel(ax, kind, data, "panel", show_legend=True)
        out = os.path.join(PANEL_DIR, f"panel_cfg_{kind}.{SAVE_FORMAT}")
        fig.savefig(out, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved panel    : {out}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  CFG Ablation Bar Chart — FM vs FM-CFG")
    print("=" * 60)

    # 加载训练集 (用于 novelty 计算)
    print("\n  Loading training set ...")
    _, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
    train_geom = info["train_geom"]

    # 加载指标 + 计算 K-NN diversity/novelty
    data = {}
    for m, _ in PAIR:
        met = load_metrics(m)
        div, nov = compute_div_nov(m, train_geom)
        data[m] = {**met, "div": div, "nov": nov}
        print(f"  [{m:<8s}] div={div:.4f}  nov={nov:.4f}")

    print()
    make_combined(data)
    print()
    make_panels(data)
    print("\nDone!")


if __name__ == "__main__":
    main()
