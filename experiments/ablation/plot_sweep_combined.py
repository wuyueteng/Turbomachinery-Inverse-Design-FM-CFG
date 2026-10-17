"""
组合敏感性分析图 — FM-CFG (3×3 grid + 9 个独立子图)
========================================================
将 p_uncond / w / N 三个 sweep 合并为单张 3×3 子图,
同时每个子图也单独输出, 方便在 Word 里手动排版.

布局 (所有列结构一致, 第 3 列同时画多样性 + 新颖性 / 双 Y 轴):
                       Col 1: R²       Col 2: nRMSE (log)    Col 3: Div + Nov
  Row 1: p_uncond     Flow/Eff/PR     Flow/Eff/PR           左 Div / 右 Nov
  Row 2: w            Flow/Eff/PR     Flow/Eff/PR           左 Div / 右 Nov
  Row 3: N            Flow/Eff/PR     Flow/Eff/PR           左 Div / 右 Nov
                      (log x)         (log x)               (log x)

定义 (与 scripts/plot_bubble.py 一致, K=5):
  - Diversity: 同一条件下 10 个生成样本之间的 K-NN 平均距离
  - Novelty:   生成样本到训练集的 K-NN 平均距离

输入: experiments/ablation/results/sweep_{p_uncond,guidance_scale,sampling_steps}.json
输出:
  experiments/ablation/figures/sweep_combined.png
  experiments/ablation/figures/panels/panel_<param>_<kind>.png  (9 个)

Usage:
    python experiments/ablation/plot_sweep_combined.py
"""

import os, sys, json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

from scripts.plot_regression import (
    SAVE_DPI, SAVE_FORMAT, SPINE_LW, TICK_DIRECTION,
    GRID_STYLE, GRID_LW, GRID_ALPHA, ANNO_BOX_EC,
    PERF_SYMBOL,
)


# ═══════════════════════════════════════════════════════════════════
#  样式配置 ── 改这里即可统一调整全图风格
# ═══════════════════════════════════════════════════════════════════

# ── 字号: combined 3×3 图 (面板紧凑) ──
COMBINED_LABEL_FONT     = 15    # X / Y 轴标题
COMBINED_TICK_FONT      = 13    # 主刻度
COMBINED_LEGEND_FONT    = 13    # 性能维度 legend
COMBINED_MARK_TEXT_FONT = 10.5  # FM-CFG 标记的竖排文字

# ── 字号: individual 子图 (与 plot_cfg_ablation.py panel 字号统一) ──
PANEL_LABEL_FONT        = 24    # X / Y 轴标题 (= 柱状图 PANEL_YLABEL_FONT / XSYMBOL / XTEXT)
PANEL_TICK_FONT         = 18    # 主刻度 (= 柱状图 PANEL_TICK_FONT)
PANEL_LEGEND_FONT       = 17    # legend (= 柱状图 PANEL_LEGEND_FONT)
PANEL_MARK_TEXT_FONT    = 16    # FM-CFG 沿线标签 (= 柱状图 PANEL_ANNO_FONT)

# ── 线条样式 ──
LINE_WIDTH              = 2.0
MARKER_SIZE             = 6.5
MARKER_SIZE_NOV         = 7.0   # Novelty 用 pentagon, 稍大一点

# ── FM-CFG 标记 (灰虚线 + 沿线标签) ──
USED_LINE_COLOR         = "#5A5A5A"
USED_LINE_WIDTH         = 1.4
USED_LINE_ALPHA         = 0.75
USED_LINE_STYLE         = "--"
USED_LABEL_BOX_FC       = "white"
USED_LABEL_BOX_EC       = "#999999"
USED_LABEL_BOX_LW       = 0.65
USED_LABEL_BOX_ALPHA    = 0.94
USED_LABEL_Y_DEFAULT    = 0.45    # 大部分面板
USED_LABEL_Y_DIVNOV     = 0.32    # divnov 双 Y 轴面板

# ── 图尺寸 ──
COMBINED_FIG_WIDTH      = 15
COMBINED_FIG_HEIGHT     = 12
PANEL_FIG_WIDTH         = 6.5     # r2 / nrmse panel 宽度
PANEL_FIG_WIDTH_DIVNOV  = 7.5     # divnov panel 加宽 (右侧 Novelty + 更舒展的绘图区)
PANEL_FIG_HEIGHT        = 5.5     # 所有 panel 等高

# ── subplot 边距 ──
COMBINED_MARGIN         = dict(left=0.06, right=0.98, top=0.96, bottom=0.06,
                               hspace=0.32, wspace=0.26)
# 按 kind 区分: divnov 是 twin-axis 需要右侧给 Novelty 标签留位; r2/nrmse 单轴无需留白
PANEL_MARGIN_DEFAULT    = dict(left=0.16, right=0.96, top=0.96, bottom=0.14)   # r2 / nrmse
PANEL_MARGIN_DIVNOV     = dict(left=0.16, right=0.84, top=0.96, bottom=0.14)   # twin-axis (右侧留 16% 给 Novelty 标签)

# ── 配色 ──
C_FLOW = "#2E5E8C"
C_EFF  = "#C0392B"
C_PR   = "#27AE60"
C_DIV  = "#16A085"   # 多样性 (teal, 左轴)
C_NOV  = "#D35400"   # 新颖性 (orange, 右轴)

# ── 多样性 / 新颖性数学符号 (与 scripts/plot_bubble.py 一致) ──
DIV_SYMBOL    = r"$D_{div}^{k}$"
NOV_SYMBOL    = r"$D_{nov}^{k}$"
DIV_AXIS_LBL  = f"Diversity  {DIV_SYMBOL}"
NOV_AXIS_LBL  = f"Novelty  {NOV_SYMBOL}"

# ── FM-CFG HPO 最佳值 ──
USED_P = 0.061
USED_W = 2.0
USED_N = 100

# ── legend 位置 ──
LEGEND_LOC_R2     = "lower right"
LEGEND_LOC_NRMSE  = "upper right"
LEGEND_LOC_DIVNOV = "best"

# ── 路径 ──
RES_DIR    = os.path.join(ROOT_DIR, "experiments", "ablation", "results")
FIG_DIR    = os.path.join(ROOT_DIR, "experiments", "ablation", "figures")
PANEL_DIR  = os.path.join(FIG_DIR, "panels")
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)


# ── rcParams 全局 ──
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth":   SPINE_LW,
    "xtick.direction":  TICK_DIRECTION,
    "ytick.direction":  TICK_DIRECTION,
    "xtick.top":        True,
    "ytick.right":      True,
    "savefig.dpi":      SAVE_DPI,
    "savefig.bbox":     "standard",   # 保持 figsize 一致, 不按内容裁剪
})


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════
def load_results(filename):
    with open(os.path.join(RES_DIR, filename)) as f:
        return json.load(f)["results"]


# ═══════════════════════════════════════════════════════════════════
#  通用绘图工具
# ═══════════════════════════════════════════════════════════════════
def mark_used(ax, x_used, label, y_axes, fs):
    """灰虚线 + 沿线竖排白盒标签."""
    ax.axvline(x_used, color=USED_LINE_COLOR, ls=USED_LINE_STYLE,
               lw=USED_LINE_WIDTH, alpha=USED_LINE_ALPHA, zorder=2.5)
    ax.text(x_used, y_axes, "  " + label,
            transform=ax.get_xaxis_transform(),
            rotation=90, ha="left", va="center", rotation_mode="anchor",
            fontsize=fs, color="#2C2C2C", family="serif",
            bbox=dict(boxstyle="round,pad=0.28",
                      fc=USED_LABEL_BOX_FC, ec=USED_LABEL_BOX_EC,
                      lw=USED_LABEL_BOX_LW, alpha=USED_LABEL_BOX_ALPHA),
            zorder=11)


def _plot_three(ax, x_vals, y_flow, y_eff, y_pr):
    l1, = ax.plot(x_vals, y_flow, "o-", color=C_FLOW, lw=LINE_WIDTH,
                  markersize=MARKER_SIZE, label=PERF_SYMBOL[0])
    l2, = ax.plot(x_vals, y_eff,  "s-", color=C_EFF,  lw=LINE_WIDTH,
                  markersize=MARKER_SIZE, label=PERF_SYMBOL[1])
    l3, = ax.plot(x_vals, y_pr,   "^-", color=C_PR,   lw=LINE_WIDTH,
                  markersize=MARKER_SIZE, label=PERF_SYMBOL[2])
    return [l1, l2, l3]


def _style(ax, fs_tick):
    ax.tick_params(labelsize=fs_tick)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════
#  单个面板绘制
# ═══════════════════════════════════════════════════════════════════
def plot_panel(ax, kind, x_vals, results,
               used_x, used_label,
               xlabel, log_x, mode,
               show_legend=False):
    """
    kind ∈ {'r2','nrmse','divnov'}
    mode ∈ {'combined', 'panel'}
    """
    # 根据 mode 选择字号
    if mode == "combined":
        fs_label     = COMBINED_LABEL_FONT
        fs_tick      = COMBINED_TICK_FONT
        fs_legend    = COMBINED_LEGEND_FONT
        fs_mark_text = COMBINED_MARK_TEXT_FONT
    else:  # "panel"
        fs_label     = PANEL_LABEL_FONT
        fs_tick      = PANEL_TICK_FONT
        fs_legend    = PANEL_LEGEND_FONT
        fs_mark_text = PANEL_MARK_TEXT_FONT

    if kind == "r2":
        handles = _plot_three(ax, x_vals,
            [r["r2_flow"] for r in results],
            [r["r2_eff"]  for r in results],
            [r["r2_pr"]   for r in results])
        ax.set_ylabel(r"$R^2$", fontsize=fs_label)
        if show_legend:
            ax.legend(handles=handles,
                      labels=[PERF_SYMBOL[0], PERF_SYMBOL[1], PERF_SYMBOL[2]],
                      prop={"size": fs_legend},
                      loc=LEGEND_LOC_R2, framealpha=0.94, edgecolor=ANNO_BOX_EC)

    elif kind == "nrmse":
        handles = _plot_three(ax, x_vals,
            [r["nrmse_flow"] for r in results],
            [r["nrmse_eff"]  for r in results],
            [r["nrmse_pr"]   for r in results])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=20))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("nRMSE", fontsize=fs_label)
        if show_legend:
            ax.legend(handles=handles,
                      labels=[PERF_SYMBOL[0], PERF_SYMBOL[1], PERF_SYMBOL[2]],
                      prop={"size": fs_legend},
                      loc=LEGEND_LOC_NRMSE, framealpha=0.94, edgecolor=ANNO_BOX_EC)

    elif kind == "divnov":
        # 双 Y 轴: 左 Diversity, 右 Novelty (符号与 plot_bubble.py 一致)
        div = [r["diversity"] for r in results]
        nov = [r["novelty"]   for r in results]
        h1, = ax.plot(x_vals, div, "v-", color=C_DIV, lw=LINE_WIDTH,
                      markersize=MARKER_SIZE, label=DIV_AXIS_LBL)
        ax.set_ylabel(DIV_AXIS_LBL, color=C_DIV, fontsize=fs_label)
        ax.tick_params(axis="y", labelcolor=C_DIV, labelsize=fs_tick)
        ax.tick_params(axis="y", which="both", right=False, labelright=False)

        ax_r = ax.twinx()
        h2, = ax_r.plot(x_vals, nov, "p-", color=C_NOV, lw=LINE_WIDTH,
                        markersize=MARKER_SIZE_NOV, label=NOV_AXIS_LBL)
        ax_r.set_ylabel(NOV_AXIS_LBL, color=C_NOV, fontsize=fs_label)
        ax_r.tick_params(axis="y", labelcolor=C_NOV, labelsize=fs_tick)
        ax_r.tick_params(axis="y", which="both", left=False)

        if show_legend:
            ax.legend(handles=[h1, h2], labels=[DIV_AXIS_LBL, NOV_AXIS_LBL],
                      prop={"size": fs_legend},
                      loc=LEGEND_LOC_DIVNOV, framealpha=0.94, edgecolor=ANNO_BOX_EC)

    # X 轴
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(int(x)) for x in x_vals])
    ax.set_xlabel(xlabel, fontsize=fs_label)
    _style(ax, fs_tick)

    # FM-CFG 标记
    y_pos = USED_LABEL_Y_DIVNOV if kind == "divnov" else USED_LABEL_Y_DEFAULT
    mark_used(ax, used_x, used_label, y_axes=y_pos, fs=fs_mark_text)


# ═══════════════════════════════════════════════════════════════════
#  整张 3×3 图
# ═══════════════════════════════════════════════════════════════════
def make_combined(rows_meta):
    fig, axes = plt.subplots(3, 3, figsize=(COMBINED_FIG_WIDTH, COMBINED_FIG_HEIGHT))
    fig.subplots_adjust(**COMBINED_MARGIN)

    for r_idx, (results, xs, used_x, used_label, xlabel, log_x) in enumerate(rows_meta):
        plot_panel(axes[r_idx, 0], "r2",     xs, results, used_x, used_label,
                   xlabel, log_x, "combined", show_legend=(r_idx == 0))
        plot_panel(axes[r_idx, 1], "nrmse",  xs, results, used_x, used_label,
                   xlabel, log_x, "combined", show_legend=(r_idx == 0))
        plot_panel(axes[r_idx, 2], "divnov", xs, results, used_x, used_label,
                   xlabel, log_x, "combined", show_legend=(r_idx == 0))

    out = os.path.join(FIG_DIR, f"sweep_combined.{SAVE_FORMAT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved combined : {out}")


# ═══════════════════════════════════════════════════════════════════
#  9 个独立子图
# ═══════════════════════════════════════════════════════════════════
PARAM_KEY = ["p_uncond", "w", "N"]
KIND_KEYS = ["r2", "nrmse", "divnov"]


def make_panels(rows_meta):
    for r_idx, (results, xs, used_x, used_label, xlabel, log_x) in enumerate(rows_meta):
        for c_idx, kind in enumerate(KIND_KEYS):
            # divnov: 加宽 + 右边距留位; r2/nrmse: 默认尺寸
            fig_w  = PANEL_FIG_WIDTH_DIVNOV if kind == "divnov" else PANEL_FIG_WIDTH
            margin = PANEL_MARGIN_DIVNOV    if kind == "divnov" else PANEL_MARGIN_DEFAULT
            fig, ax = plt.subplots(figsize=(fig_w, PANEL_FIG_HEIGHT))
            fig.subplots_adjust(**margin)
            plot_panel(ax, kind, xs, results, used_x, used_label,
                       xlabel, log_x, "panel", show_legend=True)

            fname = f"panel_{PARAM_KEY[r_idx]}_{kind}.{SAVE_FORMAT}"
            out = os.path.join(PANEL_DIR, fname)
            fig.savefig(out, dpi=SAVE_DPI)
            plt.close(fig)
            print(f"  Saved panel    : {out}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════
def main():
    p_res = load_results("sweep_p_uncond.json")
    w_res = load_results("sweep_guidance_scale.json")
    n_res = load_results("sweep_sampling_steps.json")

    p_vals = [r["p_uncond"]       for r in p_res]
    w_vals = [r["guidance_scale"] for r in w_res]
    n_vals = [r["steps"]          for r in n_res]

    rows_meta = [
        (p_res, p_vals, USED_P, r"cFM-CFG  $p_{\mathrm{uncond}}=0.061$",
         r"Conditional Dropout  $p_{\mathrm{uncond}}$", False),
        (w_res, w_vals, USED_W, r"cFM-CFG  $w=2.0$",
         r"Guidance Scale  $w$",                        False),
        (n_res, n_vals, USED_N, r"cFM-CFG  $N_{\mathrm{step}}=100$",
         r"Sampling Steps  $N_{\mathrm{step}}$",        True),
    ]

    print("=" * 60)
    print("  Combined sensitivity figure + 9 individual panels")
    print("=" * 60)
    make_combined(rows_meta)
    print()
    make_panels(rows_meta)
    print("\nDone!")


if __name__ == "__main__":
    main()
