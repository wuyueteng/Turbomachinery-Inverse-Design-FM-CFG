"""
回归散点图绘制脚本 — 读取推理结果, 绘制性能回归散点图
=====================================================
SCI 出版风格,与本项目 data/plot_data_distribution.py
及 MC Dropout 论文 plot_combined_regression_uncertainty.py 保持一致.

功能:
  1. 多模型汇总散点图 (figures/scatter_summary.png)
  2. 每个模型的单独散点图 (models/{model}/figures/)

Usage:
    python scripts/plot_regression.py                       # 全部模型
    python scripts/plot_regression.py --model ar            # 单模型
    python scripts/plot_regression.py --model ar fm_cfg     # 多模型
"""

import os, sys, argparse
import numpy as np
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════
#  全局绘图配置 — SCI 风格(对齐 MC Dropout 论文 & data_distribution)
# ═══════════════════════════════════════════════════════════════════

FONT_FAMILY       = "serif"
FONT_SERIF        = ["Times New Roman"]
MATHTEXT_FONTSET  = "stix"

# ─────────────────────────────────────────────────────────────────
#  字号控制 (改这里即可批量调整)
#  命名规则:
#    FONT_SIZE_XXX       -> 单模型独立大图 (models/{m}/figures/scatter_*.png)
#    FONT_SIZE_SUM_XXX   -> 多模型汇总图   (figures/scatter_summary.png)
#    FONT_SIZE_INSET_XXX -> 右下误差分布 inset 子图
# ─────────────────────────────────────────────────────────────────

# ── 独立大图字号(单模型) ──
FONT_SIZE_BASE       = 17   # rcParams 全局基础字号 (其它若不显式指定就回退到这里)
FONT_SIZE_TITLE      = 24   # 图标题: 例如 "AR — Isentropic Efficiency"
FONT_SIZE_LABEL      = 24   # 坐标轴标签: "Target η [-]" / "Predicted η [-]"
FONT_SIZE_TICK       = 22   # 坐标轴刻度数字: 0.83 / 0.85 / 0.88 ...
FONT_SIZE_ANNO       = 20   # 左上统计框内文字: R² / nRMSE / MAE / MSE
FONT_SIZE_LEGEND     = 15   # legend 文字 (当前已不使用 legend,留作扩展)

# ── 汇总图字号(多模型 N×3 网格,每个 panel 小,字号下调一档) ──
FONT_SIZE_SUM_TITLE  = 16   # 顶行列标题: "Mass Flow Rate $\dot{m}$ [kg/s]"
FONT_SIZE_SUM_ROW    = 14   # 左列行标签: 模型名 + "Predicted η [-]"
FONT_SIZE_SUM_LABEL  = 13   # 非边缘 panel 的 x/y 轴标签
FONT_SIZE_SUM_TICK   = 12   # 汇总图刻度数字
FONT_SIZE_SUM_ANNO   = 11   # 汇总图统计框内文字

# ── inset 误差分布子图字号 ──
FONT_SIZE_INSET_LABEL     = 19   # 独立大图 inset 的 x 轴标签: "Error ε"
FONT_SIZE_INSET_TICK      = 18   # 独立大图 inset 的刻度数字: -0.02 / 0 / 0.02
FONT_SIZE_INSET_SUM_LABEL = 10   # 汇总图 inset 的 x 轴标签
FONT_SIZE_INSET_SUM_TICK  = 9    # 汇总图 inset 的刻度数字

# ── 散点 ──
SCATTER_SIZE         = 18
SCATTER_ALPHA        = 0.55
SCATTER_SIZE_SUM     = 10
SCATTER_ALPHA_SUM    = 0.45

# ── y = x 对角线 (MC Dropout 风格: 红色虚线) ──
DIAG_COLOR           = "red"
DIAG_LW              = 1.3
DIAG_ALPHA           = 0.9
DIAG_STYLE           = "--"

# ── 标注框 ──
ANNO_BOX_FC          = "white"
ANNO_BOX_EC          = "#bdc3c7"
ANNO_BOX_ALPHA       = 0.95
ANNO_BOX_LW          = 0.7
ANNO_BOX_PAD         = 0.4

# ── 网格 ──
GRID_STYLE           = ":"
GRID_LW              = 0.5
GRID_ALPHA           = 0.4

# ── 刻度 & 边框(对齐 MC Dropout 参考) ──
TICK_DIRECTION       = "in"
TICK_LENGTH_MAJOR    = 5
TICK_LENGTH_MINOR    = 2.5
TICK_WIDTH           = 1.0
SPINE_LW             = 1.4

# ── 输出 ──
SAVE_DPI             = 600
SAVE_FORMAT          = "png"

FIG_SIZE_SINGLE      = (6.8, 6.8)
FIG_WIDTH_SUMMARY    = 15
FIG_ROW_HEIGHT       = 4.2

plt.rcParams.update({
    "font.family":          FONT_FAMILY,
    "font.serif":           FONT_SERIF,
    "font.size":            FONT_SIZE_BASE,     # ← 全局基础字号
    "mathtext.fontset":     MATHTEXT_FONTSET,
    "axes.linewidth":       SPINE_LW,
    "axes.labelsize":       FONT_SIZE_LABEL,    # ← 默认坐标轴标签字号 (单图)
    "axes.titlesize":       FONT_SIZE_TITLE,    # ← 默认图标题字号 (单图)
    "lines.linewidth":      2.0,    # 对齐 MC Dropout 参考: LINE_WIDTH_MAIN
    "xtick.labelsize":      FONT_SIZE_TICK,     # ← 默认 x 刻度字号 (单图)
    "ytick.labelsize":      FONT_SIZE_TICK,     # ← 默认 y 刻度字号 (单图)
    "xtick.direction":      TICK_DIRECTION,
    "ytick.direction":      TICK_DIRECTION,
    "xtick.major.size":     TICK_LENGTH_MAJOR,
    "ytick.major.size":     TICK_LENGTH_MAJOR,
    "xtick.major.width":    TICK_WIDTH,
    "ytick.major.width":    TICK_WIDTH,
    "xtick.minor.size":     TICK_LENGTH_MINOR,
    "ytick.minor.size":     TICK_LENGTH_MINOR,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.top":            True,
    "ytick.right":          True,
    "legend.fontsize":      FONT_SIZE_LEGEND,   # ← legend 字号 (当前未启用 legend)
    "legend.framealpha":    0.95,
    "legend.edgecolor":     ANNO_BOX_EC,
    "legend.fancybox":      False,
    "legend.handlelength":   1.4,
    "axes.labelpad":        6.0,
    "savefig.dpi":          SAVE_DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":    0.20,
})

# ═══════════════════════════════════════════════════════════════════
#  模型 & 性能维度配置
# ═══════════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR    = os.path.join(ROOT_DIR, "figures")

ALL_MODELS = ["pa_fm_cfg", "fm_cfg", "fm", "ddpm", "cinn", "ar", "cvae", "cgan", "ddpm_cfg"]
DISPLAY = {
    "pa_fm_cfg": "PA-cFM-CFG", "fm_cfg": "cFM-CFG", "fm": "FM",
    "ddpm": "cDDPM", "ddpm_cfg": "DDPM-CFG", "ar": "cAR", "cinn": "cINN",
    "cvae": "cVAE", "cgan": "cWGAN-GP",
}
COLORS = {
    "pa_fm_cfg": "#C0392B", "fm_cfg": "#8E44AD", "fm": "#E67E22",
    "ddpm": "#2C3E50", "ddpm_cfg": "#27AE60", "ar": "#2980B9", "cinn": "#7F8C8D",
    "cvae": "#16A085", "cgan": "#D35400",
}

# 统一散点配色 (SCI 风格,所有模型共用)
UNIFIED_COLOR     = "#2E5E8C"   # 深蓝灰,publication-friendly
UNIFIED_EDGE      = "white"
UNIFIED_EDGE_LW   = 0.25
INSET_FACE_ALPHA  = 0.92
INSET_EC          = "#555555"
INSET_HIST_COLOR  = "#7897B5"
INSET_CURVE_COLOR = "#1F4E79"
METRIC_X          = 0.035
METRIC_Y          = 0.965    # legend 删除后,统计框移到左上空白处

# 维度: 符号 + 单位 (MC Dropout 论文风格)
PERF_SYMBOL = {0: r"$\dot{m}$", 1: r"$\eta$", 2: r"$\pi$"}
PERF_UNIT   = {0: "[kg/s]",     1: "[-]",     2: "[-]"}
PERF_NAME   = {0: "Mass Flow Rate",
               1: "Isentropic Efficiency",
               2: "Total Pressure Ratio"}
PERF_SHORT  = {0: "flow", 1: "eff", 2: "pr"}
PERF_ORDER  = [0, 2, 1]  # 绘图顺序: Flow, PR, Eff


def axis_label(prefix, dim):
    """生成轴标签: 'Target $\\dot{m}$ [kg/s]' 风格"""
    return f"{prefix} {PERF_SYMBOL[dim]} {PERF_UNIT[dim]}"


# ═══════════════════════════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(pred, targ, dim):
    p, t = pred[:, dim], targ[:, dim]
    res = p - t
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(res)))
    mse = float(np.mean(res ** 2))
    rmse = np.sqrt(mse)
    tgt_range = float(t.max() - t.min()) + 1e-12
    nrmse = rmse / tgt_range
    # MRE: Mean Relative Error = mean(|pred - true| / |true|) × 100 [%]
    safe = np.abs(t) > 1e-9
    mre = float(np.mean(np.abs(res[safe] / t[safe])) * 100.0) if safe.any() else 0.0
    return {"r2": r2, "mae": mae, "mse": mse, "nrmse": nrmse, "mre": mre}


# ═══════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════

def load_inference_data(model_name):
    inf_dir = os.path.join(MODELS_DIR, model_name, "results", "inference")
    pred_path = os.path.join(inf_dir, "pred_performance.npy")
    targ_path = os.path.join(inf_dir, "target_performance.npy")
    if not os.path.exists(pred_path):
        return None, None
    return np.load(pred_path), np.load(targ_path)


# ═══════════════════════════════════════════════════════════════════
#  绘图辅助
# ═══════════════════════════════════════════════════════════════════

def _annotate(ax, metrics, fontsize=None):
    """Add a left-aligned metric box with a fixed equals column."""
    # 字号: 调用时不传则用 FONT_SIZE_ANNO (单图 14) / 汇总图传 FONT_SIZE_SUM_ANNO (11)
    fs = fontsize or FONT_SIZE_ANNO
    text = (f"R²    = {metrics['r2']:.4f}\n"
            f"nRMSE = {metrics['nrmse']:.4f}\n"
            f"MAE   = {metrics['mae']:.4f}\n"
            f"MRE   = {metrics['mre']:.2f}%")
    ax.text(METRIC_X, METRIC_Y, text,
            transform=ax.transAxes, fontsize=fs,
            va="top", ha="left",
            family="monospace",
            linespacing=1.28,
            bbox=dict(boxstyle=f"round,pad={ANNO_BOX_PAD}",
                      fc=ANNO_BOX_FC, alpha=ANNO_BOX_ALPHA,
                      ec=ANNO_BOX_EC, lw=ANNO_BOX_LW),
            zorder=10)


def _set_diagonal(ax, targ, pred, dim):
    """绘制 y=x 对角线 + 等比方框 + 4 个主刻度"""
    lo = min(targ[:, dim].min(), pred[:, dim].min())
    hi = max(targ[:, dim].max(), pred[:, dim].max())
    mg = (hi - lo) * 0.06
    ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg],
            color=DIAG_COLOR, ls=DIAG_STYLE, lw=DIAG_LW,
            alpha=DIAG_ALPHA, zorder=6)
    ax.set_xlim(lo - mg, hi + mg)
    ax.set_ylim(lo - mg, hi + mg)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def _style_ax(ax):
    """统一边框 + 网格"""
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW,
            alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)


def _add_legend(ax, fontsize=None):
    """Place a compact legend in the upper-left blank triangle."""
    # 字号: 不传则用 FONT_SIZE_LEGEND (15)
    fs = fontsize or FONT_SIZE_LEGEND
    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=UNIFIED_COLOR, markeredgecolor=UNIFIED_EDGE,
               markeredgewidth=UNIFIED_EDGE_LW, markersize=4.8,
               alpha=SCATTER_ALPHA, label="Prediction"),
        Line2D([0], [0], color=DIAG_COLOR, linestyle=DIAG_STYLE,
               linewidth=DIAG_LW, alpha=DIAG_ALPHA, label="1:1 line"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    prop={"size": fs}, framealpha=0.94,
                    edgecolor=ANNO_BOX_EC, facecolor="white",
                    borderpad=0.32, labelspacing=0.25,
                    handletextpad=0.45)
    leg.set_zorder(12)
    return leg


def _add_residual_inset(ax, pred, targ, dim,
                        fontsize_label=FONT_SIZE_INSET_LABEL,   # ← inset x 轴标签字号 (默认: 单图 14)
                        fontsize_tick=FONT_SIZE_INSET_TICK,     # ← inset 刻度数字字号 (默认: 单图 12)
                        bbox=(0.535, 0.1, 0.445, 0.300)):
    """Draw an inset histogram of *relative error* ε_r = (pred − true) / true × 100 [%]."""
    t = targ[:, dim]
    safe = np.abs(t) > 1e-9
    res = (pred[safe, dim] - t[safe]) / t[safe] * 100.0   # 相对误差,百分比
    ins = ax.inset_axes(bbox)

    _, bins, _ = ins.hist(
        res, bins=24, density=True,
        color=INSET_HIST_COLOR, alpha=0.78,
        edgecolor="white", linewidth=0.4,
        zorder=3,
    )

    mu = float(res.mean())
    sigma = float(res.std(ddof=1)) if res.size > 1 else 0.0
    if sigma > 1e-12:
        xs = np.linspace(bins[0], bins[-1], 200)
        pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ins.plot(xs, pdf, color=INSET_CURVE_COLOR, lw=1.6, zorder=4)

    ins.axvline(0, color=DIAG_COLOR, ls=DIAG_STYLE,
                lw=1.1, alpha=0.85, zorder=5)
    ins.axvline(mu, color="black", ls="-",
                lw=0.9, alpha=0.62, zorder=5)

    ins.set_facecolor((1.0, 1.0, 1.0, INSET_FACE_ALPHA))
    for s in ins.spines.values():
        s.set_linewidth(0.9)
        s.set_color(INSET_EC)

    ins.tick_params(axis="x", labelsize=fontsize_tick,
                    length=3.0, width=0.8, direction="in",
                    pad=3)
    ins.set_yticks([])
    ins.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ins.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ins.set_xlabel(r"RE [%]",
                   fontsize=fontsize_label,
                   labelpad=3, family="serif")
    ins.grid(False)
    ins.set_axisbelow(True)
    return ins


# ═══════════════════════════════════════════════════════════════════
#  1. 多模型汇总散点图
# ═══════════════════════════════════════════════════════════════════

def plot_summary(models_data):
    models = list(models_data.keys())
    n = len(models)
    if n == 0:
        print("  No data for summary plot.")
        return

    fig, axes = plt.subplots(n, 3,
                             figsize=(FIG_WIDTH_SUMMARY, FIG_ROW_HEIGHT * n))
    # 固定边距而非 constrained_layout,让字号变化直接反映在视觉上
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07,
                        hspace=0.35, wspace=0.25)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, m in enumerate(models):
        pred, targ = models_data[m]

        for j, d in enumerate(PERF_ORDER):
            ax = axes[i, j]
            ax.scatter(targ[:, d], pred[:, d],
                       alpha=SCATTER_ALPHA_SUM, s=SCATTER_SIZE_SUM,
                       color=UNIFIED_COLOR,
                       edgecolors=UNIFIED_EDGE, linewidths=UNIFIED_EDGE_LW,
                       zorder=3)
            _set_diagonal(ax, targ, pred, d)
            met = compute_metrics(pred, targ, d)
            _annotate(ax, met, fontsize=FONT_SIZE_SUM_ANNO)   # 汇总图统计框字号
            _add_residual_inset(ax, pred, targ, d,
                                fontsize_label=FONT_SIZE_INSET_SUM_LABEL,   # 汇总图 inset xlabel 字号
                                fontsize_tick=FONT_SIZE_INSET_SUM_TICK)     # 汇总图 inset 刻度字号
            _style_ax(ax)
            ax.tick_params(labelsize=FONT_SIZE_SUM_TICK)   # 汇总图主图刻度字号 (覆盖 rcParams)

            # 顶行: 列标题用 "符号 + 单位"
            if i == 0:
                ax.set_title(f"{PERF_NAME[d]}  {PERF_SYMBOL[d]} {PERF_UNIT[d]}",
                             fontsize=FONT_SIZE_SUM_TITLE,   # 汇总图列标题字号
                             fontweight="bold", pad=8)
            # 底行: x 标签
            if i == n - 1:
                ax.set_xlabel(axis_label("Target", d),
                              fontsize=FONT_SIZE_SUM_LABEL)   # 汇总图非首列 x 轴标签字号
            # 左列: y 标签 (模型名 + Predicted)
            if j == 0:
                ax.set_ylabel(f"{DISPLAY.get(m, m)}\nPredicted {PERF_SYMBOL[d]} {PERF_UNIT[d]}",
                              fontsize=FONT_SIZE_SUM_ROW)   # 汇总图首列 y 轴标签字号 (含模型名)
            else:
                ax.set_ylabel(axis_label("Predicted", d),
                              fontsize=FONT_SIZE_SUM_LABEL)   # 汇总图非首列 y 轴标签字号

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f"scatter_summary.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved summary: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  2. 单模型独立散点图
# ═══════════════════════════════════════════════════════════════════

def plot_individual(model_name, pred, targ):
    fig_dir = os.path.join(MODELS_DIR, model_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    display = DISPLAY.get(model_name, model_name)

    for d in range(3):
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        # 固定边距而非 constrained_layout,让字号变化直接反映在视觉上
        fig.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.13)

        ax.scatter(targ[:, d], pred[:, d],
                   alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                   color=UNIFIED_COLOR,
                   edgecolors=UNIFIED_EDGE, linewidths=UNIFIED_EDGE_LW,
                   zorder=3)
        _set_diagonal(ax, targ, pred, d)

        met = compute_metrics(pred, targ, d)
        _annotate(ax, met)                       # 统计框字号默认 FONT_SIZE_ANNO
        _add_residual_inset(ax, pred, targ, d)   # inset 字号默认 FONT_SIZE_INSET_LABEL / TICK
        _style_ax(ax)

        ax.set_title(f"{display} — {PERF_NAME[d]}",
                     fontsize=FONT_SIZE_TITLE,   # 单图主标题字号
                     fontweight="bold", pad=10)
        ax.set_xlabel(axis_label("Target",    d), fontsize=FONT_SIZE_LABEL)   # 单图 x 轴标签字号
        ax.set_ylabel(axis_label("Predicted", d), fontsize=FONT_SIZE_LABEL)   # 单图 y 轴标签字号

        fname = f"scatter_{PERF_SHORT[d]}.{SAVE_FORMAT}"
        fig.savefig(os.path.join(fig_dir, fname), dpi=SAVE_DPI)
        plt.close(fig)

    print(f"  Saved individual plots: {fig_dir}/scatter_{{flow,eff,pr}}.{SAVE_FORMAT}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="绘制回归散点图 (基于 inference.py 输出)")
    parser.add_argument("--model", nargs="*", default=None,
                        help="模型名称 (可多个), 不指定则运行全部")
    args = parser.parse_args()

    models_to_run = args.model if args.model else ALL_MODELS

    print("=" * 60)
    print("  Plot Regression Scatter Diagrams")
    print(f"  Models: {models_to_run}")
    print("=" * 60)

    models_data = {}
    for m in models_to_run:
        pred, targ = load_inference_data(m)
        if pred is None:
            print(f"  [{m}] No inference results found, skip "
                  f"(run inference.py --model {m} first)")
            continue
        models_data[m] = (pred, targ)

    if not models_data:
        print("\nNo inference data found. Run inference.py first.")
        return

    if len(models_data) > 1:
        print("\nPlotting summary scatter...")
        plot_summary(models_data)

    for m in models_data:
        print(f"\nPlotting {DISPLAY.get(m, m)}...")
        pred, targ = models_data[m]
        plot_individual(m, pred, targ)

    print("\nDone!")


if __name__ == "__main__":
    main()
