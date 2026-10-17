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
    python scripts/plot_regression.py --model ar fm_cfg     # 多模型"""
import os, sys, argparse
import numpy as np
import matplotlib.ticker as mticker
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FONT_FAMILY = 'serif'
FONT_SERIF = ['Times New Roman']
MATHTEXT_FONTSET = 'stix'
FONT_SIZE_BASE = 17
FONT_SIZE_TITLE = 24
FONT_SIZE_LABEL = 24
FONT_SIZE_TICK = 22
FONT_SIZE_ANNO = 20
FONT_SIZE_LEGEND = 15
FONT_SIZE_SUM_TITLE = 16
FONT_SIZE_SUM_ROW = 14
FONT_SIZE_SUM_LABEL = 13
FONT_SIZE_SUM_TICK = 12
FONT_SIZE_SUM_ANNO = 11
FONT_SIZE_INSET_LABEL = 19
FONT_SIZE_INSET_TICK = 18
FONT_SIZE_INSET_SUM_LABEL = 10
FONT_SIZE_INSET_SUM_TICK = 9
SCATTER_SIZE = 18
SCATTER_ALPHA = 0.55
SCATTER_SIZE_SUM = 10
SCATTER_ALPHA_SUM = 0.45
DIAG_COLOR = 'red'
DIAG_LW = 1.3
DIAG_ALPHA = 0.9
DIAG_STYLE = '--'
ANNO_BOX_FC = 'white'
ANNO_BOX_EC = '#bdc3c7'
ANNO_BOX_ALPHA = 0.95
ANNO_BOX_LW = 0.7
ANNO_BOX_PAD = 0.4
GRID_STYLE = ':'
GRID_LW = 0.5
GRID_ALPHA = 0.4
TICK_DIRECTION = 'in'
TICK_LENGTH_MAJOR = 5
TICK_LENGTH_MINOR = 2.5
TICK_WIDTH = 1.0
SPINE_LW = 1.4
SAVE_DPI = 600
SAVE_FORMAT = 'png'
FIG_SIZE_SINGLE = (6.8, 6.8)
FIG_WIDTH_SUMMARY = 15
FIG_ROW_HEIGHT = 4.2
plt.rcParams.update({'font.family': FONT_FAMILY, 'font.serif': FONT_SERIF, 'font.size': FONT_SIZE_BASE, 'mathtext.fontset': MATHTEXT_FONTSET, 'axes.linewidth': SPINE_LW, 'axes.labelsize': FONT_SIZE_LABEL, 'axes.titlesize': FONT_SIZE_TITLE, 'lines.linewidth': 2.0, 'xtick.labelsize': FONT_SIZE_TICK, 'ytick.labelsize': FONT_SIZE_TICK, 'xtick.direction': TICK_DIRECTION, 'ytick.direction': TICK_DIRECTION, 'xtick.major.size': TICK_LENGTH_MAJOR, 'ytick.major.size': TICK_LENGTH_MAJOR, 'xtick.major.width': TICK_WIDTH, 'ytick.major.width': TICK_WIDTH, 'xtick.minor.size': TICK_LENGTH_MINOR, 'ytick.minor.size': TICK_LENGTH_MINOR, 'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.top': True, 'ytick.right': True, 'legend.fontsize': FONT_SIZE_LEGEND, 'legend.framealpha': 0.95, 'legend.edgecolor': ANNO_BOX_EC, 'legend.fancybox': False, 'legend.handlelength': 1.4, 'axes.labelpad': 6.0, 'savefig.dpi': SAVE_DPI, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2})
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
FIG_DIR = os.path.join(ROOT_DIR, 'figures')
ALL_MODELS = ['fm_cfg', 'fm', 'ddpm', 'cinn', 'ar', 'cvae', 'cgan']
DISPLAY = {'fm_cfg': 'cFM-CFG', 'fm': 'FM', 'ddpm': 'cDDPM', 'ar': 'cAR', 'cinn': 'cINN', 'cvae': 'cVAE', 'cgan': 'cWGAN-GP'}
COLORS = {'fm_cfg': '#8E44AD', 'fm': '#E67E22', 'ddpm': '#2C3E50', 'ar': '#2980B9', 'cinn': '#7F8C8D', 'cvae': '#16A085', 'cgan': '#D35400'}
UNIFIED_COLOR = '#2E5E8C'
UNIFIED_EDGE = 'white'
UNIFIED_EDGE_LW = 0.25
INSET_FACE_ALPHA = 0.92
INSET_EC = '#555555'
INSET_HIST_COLOR = '#7897B5'
INSET_CURVE_COLOR = '#1F4E79'
METRIC_X = 0.035
METRIC_Y = 0.965
PERF_SYMBOL = {0: '$\\dot{m}$', 1: '$\\eta$', 2: '$\\pi$'}
PERF_UNIT = {0: '[kg/s]', 1: '[-]', 2: '[-]'}
PERF_NAME = {0: 'Mass Flow Rate', 1: 'Isentropic Efficiency', 2: 'Total Pressure Ratio'}
PERF_SHORT = {0: 'flow', 1: 'eff', 2: 'pr'}
PERF_ORDER = [0, 2, 1]

def axis_label(prefix, dim):
    """生成轴标签: 'Target $\\dot{m}$ [kg/s]' 风格"""
    return f'{prefix} {PERF_SYMBOL[dim]} {PERF_UNIT[dim]}'

def compute_metrics(pred, targ, dim):
    (p, t) = (pred[:, dim], targ[:, dim])
    res = p - t
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(res)))
    mse = float(np.mean(res ** 2))
    rmse = np.sqrt(mse)
    tgt_range = float(t.max() - t.min()) + 1e-12
    nrmse = rmse / tgt_range
    safe = np.abs(t) > 1e-09
    mre = float(np.mean(np.abs(res[safe] / t[safe])) * 100.0) if safe.any() else 0.0
    return {'r2': r2, 'mae': mae, 'mse': mse, 'nrmse': nrmse, 'mre': mre}

def load_inference_data(model_name):
    inf_dir = os.path.join(MODELS_DIR, model_name, 'results', 'inference')
    pred_path = os.path.join(inf_dir, 'pred_performance.npy')
    targ_path = os.path.join(inf_dir, 'target_performance.npy')
    if not os.path.exists(pred_path):
        return (None, None)
    return (np.load(pred_path), np.load(targ_path))

def _annotate(ax, metrics, fontsize=None):
    """Add a left-aligned metric box with a fixed equals column."""
    fs = fontsize or FONT_SIZE_ANNO
    text = f"R²    = {metrics['r2']:.4f}\nnRMSE = {metrics['nrmse']:.4f}\nMAE   = {metrics['mae']:.4f}\nMRE   = {metrics['mre']:.2f}%"
    ax.text(METRIC_X, METRIC_Y, text, transform=ax.transAxes, fontsize=fs, va='top', ha='left', family='monospace', linespacing=1.28, bbox=dict(boxstyle=f'round,pad={ANNO_BOX_PAD}', fc=ANNO_BOX_FC, alpha=ANNO_BOX_ALPHA, ec=ANNO_BOX_EC, lw=ANNO_BOX_LW), zorder=10)

def _set_diagonal(ax, targ, pred, dim):
    """绘制 y=x 对角线 + 等比方框 + 4 个主刻度"""
    lo = min(targ[:, dim].min(), pred[:, dim].min())
    hi = max(targ[:, dim].max(), pred[:, dim].max())
    mg = (hi - lo) * 0.06
    ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], color=DIAG_COLOR, ls=DIAG_STYLE, lw=DIAG_LW, alpha=DIAG_ALPHA, zorder=6)
    ax.set_xlim(lo - mg, hi + mg)
    ax.set_ylim(lo - mg, hi + mg)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

def _style_ax(ax):
    """统一边框 + 网格"""
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.grid(True, linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)

def _add_legend(ax, fontsize=None):
    """Place a compact legend in the upper-left blank triangle."""
    fs = fontsize or FONT_SIZE_LEGEND
    handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=UNIFIED_COLOR, markeredgecolor=UNIFIED_EDGE, markeredgewidth=UNIFIED_EDGE_LW, markersize=4.8, alpha=SCATTER_ALPHA, label='Prediction'), Line2D([0], [0], color=DIAG_COLOR, linestyle=DIAG_STYLE, linewidth=DIAG_LW, alpha=DIAG_ALPHA, label='1:1 line')]
    leg = ax.legend(handles=handles, loc='upper left', prop={'size': fs}, framealpha=0.94, edgecolor=ANNO_BOX_EC, facecolor='white', borderpad=0.32, labelspacing=0.25, handletextpad=0.45)
    leg.set_zorder(12)
    return leg

def _add_residual_inset(ax, pred, targ, dim, fontsize_label=FONT_SIZE_INSET_LABEL, fontsize_tick=FONT_SIZE_INSET_TICK, bbox=(0.535, 0.1, 0.445, 0.3)):
    """Draw an inset histogram of *relative error* ε_r = (pred − true) / true × 100 [%]."""
    t = targ[:, dim]
    safe = np.abs(t) > 1e-09
    res = (pred[safe, dim] - t[safe]) / t[safe] * 100.0
    ins = ax.inset_axes(bbox)
    (_, bins, _) = ins.hist(res, bins=24, density=True, color=INSET_HIST_COLOR, alpha=0.78, edgecolor='white', linewidth=0.4, zorder=3)
    mu = float(res.mean())
    sigma = float(res.std(ddof=1)) if res.size > 1 else 0.0
    if sigma > 1e-12:
        xs = np.linspace(bins[0], bins[-1], 200)
        pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ins.plot(xs, pdf, color=INSET_CURVE_COLOR, lw=1.6, zorder=4)
    ins.axvline(0, color=DIAG_COLOR, ls=DIAG_STYLE, lw=1.1, alpha=0.85, zorder=5)
    ins.axvline(mu, color='black', ls='-', lw=0.9, alpha=0.62, zorder=5)
    ins.set_facecolor((1.0, 1.0, 1.0, INSET_FACE_ALPHA))
    for s in ins.spines.values():
        s.set_linewidth(0.9)
        s.set_color(INSET_EC)
    ins.tick_params(axis='x', labelsize=fontsize_tick, length=3.0, width=0.8, direction='in', pad=3)
    ins.set_yticks([])
    ins.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ins.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ins.set_xlabel('RE [%]', fontsize=fontsize_label, labelpad=3, family='serif')
    ins.grid(False)
    ins.set_axisbelow(True)
    return ins

def plot_summary(models_data):
    models = list(models_data.keys())
    n = len(models)
    if n == 0:
        print('  No data for summary plot.')
        return
    (fig, axes) = plt.subplots(n, 3, figsize=(FIG_WIDTH_SUMMARY, FIG_ROW_HEIGHT * n))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07, hspace=0.35, wspace=0.25)
    if n == 1:
        axes = axes[np.newaxis, :]
    for (i, m) in enumerate(models):
        (pred, targ) = models_data[m]
        for (j, d) in enumerate(PERF_ORDER):
            ax = axes[i, j]
            ax.scatter(targ[:, d], pred[:, d], alpha=SCATTER_ALPHA_SUM, s=SCATTER_SIZE_SUM, color=UNIFIED_COLOR, edgecolors=UNIFIED_EDGE, linewidths=UNIFIED_EDGE_LW, zorder=3)
            _set_diagonal(ax, targ, pred, d)
            met = compute_metrics(pred, targ, d)
            _annotate(ax, met, fontsize=FONT_SIZE_SUM_ANNO)
            _add_residual_inset(ax, pred, targ, d, fontsize_label=FONT_SIZE_INSET_SUM_LABEL, fontsize_tick=FONT_SIZE_INSET_SUM_TICK)
            _style_ax(ax)
            ax.tick_params(labelsize=FONT_SIZE_SUM_TICK)
            if i == 0:
                ax.set_title(f'{PERF_NAME[d]}  {PERF_SYMBOL[d]} {PERF_UNIT[d]}', fontsize=FONT_SIZE_SUM_TITLE, fontweight='bold', pad=8)
            if i == n - 1:
                ax.set_xlabel(axis_label('Target', d), fontsize=FONT_SIZE_SUM_LABEL)
            if j == 0:
                ax.set_ylabel(f'{DISPLAY.get(m, m)}\nPredicted {PERF_SYMBOL[d]} {PERF_UNIT[d]}', fontsize=FONT_SIZE_SUM_ROW)
            else:
                ax.set_ylabel(axis_label('Predicted', d), fontsize=FONT_SIZE_SUM_LABEL)
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f'scatter_summary.{SAVE_FORMAT}')
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f'  Saved summary: {out_path}')

def plot_individual(model_name, pred, targ):
    fig_dir = os.path.join(MODELS_DIR, model_name, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    display = DISPLAY.get(model_name, model_name)
    for d in range(3):
        (fig, ax) = plt.subplots(figsize=FIG_SIZE_SINGLE)
        fig.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.13)
        ax.scatter(targ[:, d], pred[:, d], alpha=SCATTER_ALPHA, s=SCATTER_SIZE, color=UNIFIED_COLOR, edgecolors=UNIFIED_EDGE, linewidths=UNIFIED_EDGE_LW, zorder=3)
        _set_diagonal(ax, targ, pred, d)
        met = compute_metrics(pred, targ, d)
        _annotate(ax, met)
        _add_residual_inset(ax, pred, targ, d)
        _style_ax(ax)
        ax.set_title(f'{display} — {PERF_NAME[d]}', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=10)
        ax.set_xlabel(axis_label('Target', d), fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(axis_label('Predicted', d), fontsize=FONT_SIZE_LABEL)
        fname = f'scatter_{PERF_SHORT[d]}.{SAVE_FORMAT}'
        fig.savefig(os.path.join(fig_dir, fname), dpi=SAVE_DPI)
        plt.close(fig)
    print(f'  Saved individual plots: {fig_dir}/scatter_{{flow,eff,pr}}.{SAVE_FORMAT}')

def main():
    parser = argparse.ArgumentParser(description='绘制回归散点图 (基于 inference.py 输出)')
    parser.add_argument('--model', nargs='*', default=None, help='模型名称 (可多个), 不指定则运行全部')
    args = parser.parse_args()
    models_to_run = args.model if args.model else ALL_MODELS
    print('=' * 60)
    print('  Plot Regression Scatter Diagrams')
    print(f'  Models: {models_to_run}')
    print('=' * 60)
    models_data = {}
    for m in models_to_run:
        (pred, targ) = load_inference_data(m)
        if pred is None:
            print(f'  [{m}] No inference results found, skip (run inference.py --model {m} first)')
            continue
        models_data[m] = (pred, targ)
    if not models_data:
        print('\nNo inference data found. Run inference.py first.')
        return
    if len(models_data) > 1:
        print('\nPlotting summary scatter...')
        plot_summary(models_data)
    for m in models_data:
        print(f'\nPlotting {DISPLAY.get(m, m)}...')
        (pred, targ) = models_data[m]
        plot_individual(m, pred, targ)
    print('\nDone!')
if __name__ == '__main__':
    main()
