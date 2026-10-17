"""
设计参数概率密度分布图 — 21 个几何参数的 KDE 分布对比
=====================================================
读取 inference.py 的输出 + 真实测试集数据, 绘制:
  1. 组图 (3 sections x 7 params): figures/kde_distribution.png
  2. 每个参数单独图: models/{model}/figures/kde_{section}_{param}.png

Usage:
    python scripts/plot_distribution.py                           # 全部模型
    python scripts/plot_distribution.py --model ar                # 单模型
    python scripts/plot_distribution.py --model ar fm_cfg  # 多个模型"""
PLOT_MODELS = ['fm_cfg', 'ddpm', 'cinn', 'ar', 'cvae', 'cgan']
import os, sys, argparse, json, csv
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.data_utils import get_dataloaders
FONT_FAMILY = 'serif'
FONT_SERIF = ['Times New Roman']
MATHTEXT_FONTSET = 'stix'
FONT_SIZE_BASE = 16
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 21
FONT_SIZE_SECTION = 20
FONT_SIZE_SINGLE_TITLE = 15
FONT_SIZE_SINGLE_LABEL = 14
FONT_SIZE_SINGLE_TICK = 12
FONT_SIZE_SINGLE_LEGEND = 10
KDE_BW_METHOD = 0.08
KDE_GRID_POINTS = 300
REAL_COLOR = '#1A1A1A'
REAL_LW = 2.0
REAL_FILL_ALPHA = 0.15
GEN_LW = 1.3
GEN_ALPHA = 0.85
TICK_DIRECTION = 'in'
TICK_LENGTH = 4
TICK_WIDTH = 0.8
SPINE_LW = 0.8
SAVE_DPI = 600
SAVE_FORMAT = 'png'
FIG_SIZE_GRID = (22, 9)
FIG_SIZE_SINGLE = (5.5, 4.0)
plt.rcParams.update({'font.family': FONT_FAMILY, 'font.serif': FONT_SERIF, 'font.size': FONT_SIZE_BASE, 'axes.labelsize': FONT_SIZE_LABEL, 'axes.titlesize': FONT_SIZE_TITLE, 'axes.linewidth': SPINE_LW, 'xtick.labelsize': FONT_SIZE_TICK, 'ytick.labelsize': FONT_SIZE_TICK, 'xtick.direction': TICK_DIRECTION, 'ytick.direction': TICK_DIRECTION, 'xtick.major.size': TICK_LENGTH, 'ytick.major.size': TICK_LENGTH, 'xtick.major.width': TICK_WIDTH, 'ytick.major.width': TICK_WIDTH, 'mathtext.fontset': MATHTEXT_FONTSET, 'savefig.dpi': SAVE_DPI, 'savefig.bbox': 'tight'})
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
FIG_DIR = os.path.join(ROOT_DIR, 'figures')
ALL_MODELS = ['fm_cfg', 'fm', 'ddpm', 'cinn', 'ar', 'cvae', 'cgan']
DISPLAY = {'fm_cfg': 'cFM-CFG', 'fm': 'FM', 'ddpm': 'cDDPM', 'ar': 'cAR', 'cinn': 'cINN', 'cvae': 'cVAE', 'cgan': 'cWGAN-GP'}
LEGEND_ORDER = ['fm_cfg', 'ar', 'cvae', 'cgan', 'ddpm', 'cinn']
COLORS = {'fm_cfg': '#8E44AD', 'fm': '#E67E22', 'ddpm': '#2C3E50', 'ar': '#2980B9', 'cinn': '#7F8C8D', 'cvae': '#16A085', 'cgan': '#D35400'}
PARAM_LABELS = ['$\\beta_{1k}$', '$\\beta_{2k}$', '$C$', '$T$', '$TP$', '$RS$', '$RB$']
PARAM_SAFE = ['beta1k', 'beta2k', 'C', 'TmaxCH', 'TmaxP', 'SWA', 'BOWA']
SECTIONS = ['Hub', 'Mid', 'Tip']

def load_test_geom():
    """加载归一化测试集几何参数"""
    (_, _, _, info) = get_dataloaders(batch_size=256, use_merged=True)
    return info['test_geom']

def load_gen_geom(model_name):
    """从 inference 输出加载归一化生成几何参数"""
    path = os.path.join(MODELS_DIR, model_name, 'results', 'inference', 'geometry_norm.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)

def resolve_models(cli_models):
    """Resolve which models to plot, preferring CLI over in-file config."""
    if cli_models:
        models = cli_models
        source = 'CLI --model'
    elif PLOT_MODELS:
        models = PLOT_MODELS
        source = 'PLOT_MODELS'
    else:
        models = ALL_MODELS
        source = 'ALL_MODELS'
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f'Unknown model(s): {unknown}. Available models: {ALL_MODELS}')
    return (list(dict.fromkeys(models)), source)

def _safe_kde(values, x_grid):
    """安全计算 KDE, 数据不足或方差为零时返回 None"""
    values = values[np.isfinite(values) & (values >= -0.5) & (values <= 1.5)]
    if len(values) < 10 or np.std(values) < 1e-08:
        return None
    try:
        kde = gaussian_kde(values, bw_method=KDE_BW_METHOD)
        return kde(x_grid)
    except Exception:
        return None

def _js_divergence(real_vals, gen_vals, n_bins=100):
    """基于直方图离散化计算 JS 散度 (对称, 有界 [0,1])"""
    lo = min(real_vals.min(), gen_vals.min()) - 1e-06
    hi = max(real_vals.max(), gen_vals.max()) + 1e-06
    bins = np.linspace(lo, hi, n_bins + 1)
    (p, _) = np.histogram(real_vals, bins=bins, density=False)
    (q, _) = np.histogram(gen_vals, bins=bins, density=False)
    p = p.astype(float) / (p.sum() + 1e-12)
    q = q.astype(float) / (q.sum() + 1e-12)
    return float(jensenshannon(p, q) ** 2)

def compute_similarity(test_geom, gen_data):
    """
    对每个模型计算 21 个参数的分布相似度, 返回汇总结果.

    指标:
      - Wasserstein Distance (EMD): 推土机距离, 越小越好
      - KS Statistic: Kolmogorov-Smirnov 统计量, 越小越好
      - JS Divergence: Jensen-Shannon 散度, 越小越好
    """
    results = {}
    for (m, gen_geom) in gen_data.items():
        (wd_list, ks_list, js_list) = ([], [], [])
        param_details = []
        for dim_idx in range(21):
            (sec, var) = (dim_idx // 7, dim_idx % 7)
            param_name = f'{SECTIONS[sec]}_{PARAM_SAFE[var]}'
            real = test_geom[:, dim_idx]
            gen = gen_geom[:, dim_idx]
            gen = gen[np.isfinite(gen)]
            wd = wasserstein_distance(real, gen)
            (ks_stat, _) = ks_2samp(real, gen)
            js = _js_divergence(real, gen)
            wd_list.append(wd)
            ks_list.append(ks_stat)
            js_list.append(js)
            param_details.append({'param': param_name, 'wd': wd, 'ks': ks_stat, 'js': js})
        results[m] = {'params': param_details, 'wd_mean': float(np.mean(wd_list)), 'ks_mean': float(np.mean(ks_list)), 'js_mean': float(np.mean(js_list))}
    return results

def print_ranking(sim_results):
    """打印分布相似度排名表"""
    models = list(sim_results.keys())
    if not models:
        return ([], {})
    wd_rank = sorted(models, key=lambda m: sim_results[m]['wd_mean'])
    ks_rank = sorted(models, key=lambda m: sim_results[m]['ks_mean'])
    js_rank = sorted(models, key=lambda m: sim_results[m]['js_mean'])
    rank_scores = {}
    for m in models:
        rank_scores[m] = wd_rank.index(m) + ks_rank.index(m) + js_rank.index(m)
    overall_rank = sorted(models, key=lambda m: rank_scores[m])
    print('\n' + '=' * 90)
    print('  Distribution Similarity Ranking (lower = more similar to real data)')
    print('=' * 90)
    header = f"  {'Rank':<5s} {'Model':<12s} {'Wasserstein':>12s} {'KS Stat':>12s} {'JS Div':>12s} {'Score':>7s}"
    print(header)
    print('  ' + '-' * 86)
    for (rank, m) in enumerate(overall_rank, 1):
        r = sim_results[m]
        display = DISPLAY.get(m, m)
        wd_r = wd_rank.index(m) + 1
        ks_r = ks_rank.index(m) + 1
        js_r = js_rank.index(m) + 1
        print(f"  {rank:<5d} {display:<12s} {r['wd_mean']:>10.4f}({wd_r}) {r['ks_mean']:>10.4f}({ks_r}) {r['js_mean']:>10.4f}({js_r}) {rank_scores[m]:>5d}")
    print('=' * 90)
    best = overall_rank[0]
    print(f'\n  Best model: {DISPLAY.get(best, best)}')
    print(f"  {'Parameter':<14s} {'Wasserstein':>12s} {'KS Stat':>12s} {'JS Div':>12s}")
    print('  ' + '-' * 54)
    for p in sim_results[best]['params']:
        print(f"  {p['param']:<14s} {p['wd']:>12.4f} {p['ks']:>12.4f} {p['js']:>12.4f}")
    return (overall_rank, rank_scores)

def save_similarity(sim_results, overall_rank, rank_scores, out_dir):
    """保存相似度结果为 txt + json"""
    os.makedirs(out_dir, exist_ok=True)
    json_data = {}
    for m in sim_results:
        r = sim_results[m]
        json_data[m] = {'display': DISPLAY.get(m, m), 'wd_mean': r['wd_mean'], 'ks_mean': r['ks_mean'], 'js_mean': r['js_mean'], 'rank_score': rank_scores[m], 'overall_rank': overall_rank.index(m) + 1, 'params': r['params']}
    with open(os.path.join(out_dir, 'distribution_similarity.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    txt_path = os.path.join(out_dir, 'distribution_similarity.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Distribution Similarity Ranking\n')
        f.write('(lower = more similar to real data)\n')
        f.write('=' * 75 + '\n\n')
        f.write(f"{'Rank':<5s} {'Model':<12s} {'Wasserstein':>12s} {'KS Stat':>12s} {'JS Div':>12s} {'Score':>7s}\n")
        f.write('-' * 75 + '\n')
        for (rank, m) in enumerate(overall_rank, 1):
            r = sim_results[m]
            display = DISPLAY.get(m, m)
            f.write(f"{rank:<5d} {display:<12s} {r['wd_mean']:>12.4f} {r['ks_mean']:>12.4f} {r['js_mean']:>12.4f} {rank_scores[m]:>5d}\n")
        f.write('=' * 75 + '\n\n')
        for m in overall_rank:
            display = DISPLAY.get(m, m)
            f.write(f'\n--- {display} ---\n')
            f.write(f"{'Parameter':<14s} {'Wasserstein':>12s} {'KS Stat':>12s} {'JS Div':>12s}\n")
            f.write('-' * 54 + '\n')
            for p in sim_results[m]['params']:
                f.write(f"{p['param']:<14s} {p['wd']:>12.4f} {p['ks']:>12.4f} {p['js']:>12.4f}\n")
    csv_path = os.path.join(out_dir, 'distribution_similarity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Parameter', 'Wasserstein', 'KS_Stat', 'JS_Div'])
        for m in overall_rank:
            display = DISPLAY.get(m, m)
            for p in sim_results[m]['params']:
                writer.writerow([display, p['param'], f"{p['wd']:.6f}", f"{p['ks']:.6f}", f"{p['js']:.6f}"])
    print(f'\n  Saved similarity results: {out_dir}/distribution_similarity.{{json,txt,csv}}')

def plot_grid(test_geom, gen_data):
    """3x7 组图: 行=截面 (Tip/Mid/Hub 从上到下), 列=参数"""
    x_grid = np.linspace(0, 1, KDE_GRID_POINTS)
    models = list(gen_data.keys())
    (fig, axes) = plt.subplots(3, 7, figsize=FIG_SIZE_GRID)
    registered_labels = set()
    for sec in range(3):
        row = 2 - sec
        for var in range(7):
            dim_idx = sec * 7 + var
            ax = axes[row, var]
            real_vals = test_geom[:, dim_idx]
            real_kde = _safe_kde(real_vals, x_grid)
            if real_kde is not None:
                lbl = 'Real' if 'Real' not in registered_labels else None
                ax.fill_between(x_grid, real_kde, alpha=REAL_FILL_ALPHA, color=REAL_COLOR)
                ax.plot(x_grid, real_kde, color=REAL_COLOR, lw=REAL_LW, label=lbl)
                if lbl:
                    registered_labels.add(lbl)
            for m in models:
                gen_vals = gen_data[m][:, dim_idx]
                gen_kde = _safe_kde(gen_vals, x_grid)
                if gen_kde is not None:
                    disp = DISPLAY.get(m, m)
                    lbl = disp if disp not in registered_labels else None
                    ax.plot(x_grid, gen_kde, color=COLORS.get(m, '#666'), lw=GEN_LW, alpha=GEN_ALPHA, label=lbl)
                    if lbl:
                        registered_labels.add(lbl)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([])
            ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
            if row == 0:
                ax.set_title(PARAM_LABELS[var], fontsize=FONT_SIZE_TITLE, pad=8)
            if var == 0:
                ax.set_ylabel(SECTIONS[sec], fontsize=FONT_SIZE_SECTION, fontweight='bold')
            if row == 2:
                ax.set_xlabel('Normalized value', fontsize=FONT_SIZE_LABEL)
    label_to_handle = {}
    for ax_row in axes:
        for ax in ax_row:
            (h, l) = ax.get_legend_handles_labels()
            for (hi, li) in zip(h, l):
                if li not in label_to_handle:
                    label_to_handle[li] = hi
    ordered_labels = []
    if 'Real' in label_to_handle:
        ordered_labels.append('Real')
    for mk in LEGEND_ORDER:
        disp = DISPLAY.get(mk, mk)
        if disp in label_to_handle and disp not in ordered_labels:
            ordered_labels.append(disp)
    for mk in ALL_MODELS:
        disp = DISPLAY.get(mk, mk)
        if disp in label_to_handle and disp not in ordered_labels:
            ordered_labels.append(disp)
    all_labels = ordered_labels
    all_handles = [label_to_handle[l] for l in ordered_labels]
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', ncol=min(len(all_handles), 9), prop={'size': FONT_SIZE_LEGEND}, framealpha=0.92, edgecolor='#cccccc', bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.subplots_adjust(top=0.9)
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f'kde_distribution.{SAVE_FORMAT}')
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f'  Saved grid: {out_path}')

def plot_individual(test_geom, gen_data):
    """为每个模型 x 每个参数 生成独立 KDE 图"""
    x_grid = np.linspace(0, 1, KDE_GRID_POINTS)
    _saved_rc = {k: plt.rcParams[k] for k in ['font.size', 'axes.labelsize', 'axes.titlesize', 'xtick.labelsize', 'ytick.labelsize']}
    plt.rcParams.update({'font.size': FONT_SIZE_SINGLE_TICK, 'axes.labelsize': FONT_SIZE_SINGLE_LABEL, 'axes.titlesize': FONT_SIZE_SINGLE_TITLE, 'xtick.labelsize': FONT_SIZE_SINGLE_TICK, 'ytick.labelsize': FONT_SIZE_SINGLE_TICK})
    for (m, gen_geom) in gen_data.items():
        fig_dir = os.path.join(MODELS_DIR, m, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        display = DISPLAY.get(m, m)
        color = COLORS.get(m, '#2980B9')
        for sec in range(3):
            for var in range(7):
                dim_idx = sec * 7 + var
                (fig, ax) = plt.subplots(figsize=FIG_SIZE_SINGLE)
                real_vals = test_geom[:, dim_idx]
                real_kde = _safe_kde(real_vals, x_grid)
                if real_kde is not None:
                    ax.fill_between(x_grid, real_kde, alpha=REAL_FILL_ALPHA, color=REAL_COLOR)
                    ax.plot(x_grid, real_kde, color=REAL_COLOR, lw=REAL_LW, label='Real')
                gen_vals = gen_geom[:, dim_idx]
                gen_kde = _safe_kde(gen_vals, x_grid)
                if gen_kde is not None:
                    ax.fill_between(x_grid, gen_kde, alpha=0.1, color=color)
                    ax.plot(x_grid, gen_kde, color=color, lw=GEN_LW + 0.3, alpha=GEN_ALPHA, label=display)
                ax.set_xlim(0, 1)
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([])
                ax.set_xlabel('Normalized value', fontsize=FONT_SIZE_SINGLE_LABEL)
                ax.set_ylabel('Density', fontsize=FONT_SIZE_SINGLE_LABEL)
                ax.set_title(f'{display}  —  {SECTIONS[sec]} {PARAM_LABELS[var]}', fontsize=FONT_SIZE_SINGLE_TITLE, fontweight='medium', pad=10)
                ax.tick_params(labelsize=FONT_SIZE_SINGLE_TICK)
                ax.legend(fontsize=FONT_SIZE_SINGLE_LEGEND, loc='upper right', framealpha=0.92, edgecolor='#cccccc')
                plt.tight_layout()
                fname = f'kde_{SECTIONS[sec]}_{PARAM_SAFE[var]}.{SAVE_FORMAT}'
                fig.savefig(os.path.join(fig_dir, fname), dpi=SAVE_DPI)
                plt.close(fig)
        print(f'  Saved 21 KDE plots: {fig_dir}/kde_{{section}}_{{param}}.{SAVE_FORMAT}')
    plt.rcParams.update(_saved_rc)

def main():
    parser = argparse.ArgumentParser(description='绘制 21 个设计参数的概率密度分布图')
    parser.add_argument('--model', nargs='*', default=None, help='模型名称 (可多个), 不指定则运行全部')
    parser.add_argument('--no-individual', action='store_true', help='跳过单参数独立图 (只画组图)')
    args = parser.parse_args()
    (models_to_run, model_source) = resolve_models(args.model)
    print('=' * 60)
    print('  Plot Parameter Distribution (KDE)')
    print(f'  Models: {models_to_run}')
    print(f'  Model source: {model_source}')
    print('=' * 60)
    print('\nLoading test set geometry...')
    test_geom = load_test_geom()
    print(f'  Test set shape: {test_geom.shape}')
    gen_data = {}
    for m in models_to_run:
        geom = load_gen_geom(m)
        if geom is None:
            print(f'  [{m}] No inference results, skip (run inference.py --model {m} first)')
            continue
        gen_data[m] = geom
        print(f'  [{m}] Loaded {geom.shape[0]} samples')
    if not gen_data:
        print('\nNo generated data found. Run inference.py first.')
        return
    print('\nComputing distribution similarity...')
    sim_results = compute_similarity(test_geom, gen_data)
    (overall_rank, rank_scores) = print_ranking(sim_results)
    save_similarity(sim_results, overall_rank, rank_scores, FIG_DIR)
    print('\nPlotting 3x7 grid...')
    plot_grid(test_geom, gen_data)
    if not args.no_individual:
        print('\nPlotting individual KDE plots...')
        plot_individual(test_geom, gen_data)
    print('\nDone!')
if __name__ == '__main__':
    main()
