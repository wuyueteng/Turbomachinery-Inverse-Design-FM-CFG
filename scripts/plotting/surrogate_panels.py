from pathlib import Path
"""
回归散点 + MC Dropout 校准后置信带 (无边缘 KDE 版)

每个 panel 同时展示:
  - 训练/测试集散点 (Train/Test scatter)  — 背景化,不抢戏
  - ±1σ / ±2σ 置信带 (calibrated)         — 主体,饱和清晰
  - 分箱预测均值曲线 binned μ(x)
  - 综合统计指标 (R², nRMSE, MAE, τ, Coverage@1σ, Coverage@2σ, ρ)

输出:
  - 3 张独立 combined panel
  - 1 张 combined 1x3 行图
  - 1 张 2x3 替代版 Figure 8 (combined row + robustness row)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =================================================================
# 字体大小配置 (统一在这里改)
# =================================================================
FONT_BASE          = 17
FONT_AXES_LABEL    = 21
FONT_AXES_TITLE    = 22
FONT_TICKS         = 16
FONT_LEGEND        = 15
FONT_STATS_BOX     = 14
FONT_CBAR_LABEL    = 16
FONT_CBAR_TICK     = 14
FONT_MEAN_ANNOT    = 14

LINE_WIDTH_AXIS    = 1.4
LINE_WIDTH_MAIN    = 2.0

# 整图尺寸 (英寸) — 字号变大后同步放大画布,防止拥挤
FIG_SIZE_SINGLE    = (6.5, 6.5)
FIG_SIZE_ROW       = (20, 6.8)
FIG_SIZE_2x3       = (22, 13.5)

# 视觉参数 — 散点背景化 + 置信带主体化
SCATTER_TRAIN_SIZE   = 15
SCATTER_TRAIN_ALPHA  = 0.22
SCATTER_TRAIN_COLOR  = '#9ea7b3'

SCATTER_TEST_SIZE    = 22
SCATTER_TEST_ALPHA   = 0.65
SCATTER_TEST_COLOR   = '#2ca25f'

BAND_2SIGMA_COLOR    = '#5dade2'   # 浅蓝
BAND_2SIGMA_ALPHA    = 0.5
BAND_1SIGMA_COLOR    = '#1f618d'   # 深蓝
BAND_1SIGMA_ALPHA    = 0.5

MU_LINE_COLOR        = '#0b2545'   # 极深海蓝,与带子拉开
MU_LINE_WIDTH        = 2.2

# 平滑置信带参数 — 滑动窗口覆盖完整数据区间
BAND_QUERY_POINTS    = 100   # 沿 y_true 方向的查询点数 (band 越平滑,越大)
BAND_WINDOW_FRAC     = 0.20  # 每个查询点用最近 20% 样本估计 μ, σ

# =================================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': FONT_BASE,
    'axes.linewidth': LINE_WIDTH_AXIS,
    'axes.labelsize': FONT_AXES_LABEL,
    'axes.titlesize': FONT_AXES_TITLE,
    'xtick.labelsize': FONT_TICKS,
    'ytick.labelsize': FONT_TICKS,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'xtick.top': True,
    'ytick.right': True,
    'legend.fontsize': FONT_LEGEND,
    'lines.linewidth': LINE_WIDTH_MAIN,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})

# =================================================================
# Paths
# =================================================================
HERE = str(Path(__file__).resolve().parents[2] / 'paper_results/surrogate')
ROBUST_DIR = HERE
PLOT_DIR = os.path.dirname(os.path.dirname(HERE)) + r"\论文绘图"

TARGETS = [
    {'idx': 0, 'name': 'Mass Flow Rate',        'symbol': r'$\dot{m}$', 'unit': ' [kg/s]',
     'fname': 'combined_mass_flow_rate',
     'r2_col': 'Mass_R2', 'nrmse_col': 'Mass_nRMSE', 'mae_col': 'Mass_MAE', 'cbar_label': 'MAE [kg/s]'},
    {'idx': 1, 'name': 'Isentropic Efficiency', 'symbol': r'$\eta$',    'unit': ' [-]',
     'fname': 'combined_isentropic_efficiency',
     'r2_col': 'Eff_R2', 'nrmse_col': 'Eff_nRMSE', 'mae_col': 'Eff_MAE', 'cbar_label': 'MAE [-]'},
    {'idx': 2, 'name': 'Total Pressure Ratio', 'symbol': r'$\pi$',     'unit': ' [-]',
     'fname': 'combined_total_pressure_ratio',
     'r2_col': 'Pre_R2', 'nrmse_col': 'Pre_nRMSE', 'mae_col': 'Pre_MAE', 'cbar_label': 'MAE [-]'},
]

# =================================================================
# Load data
# =================================================================
# Robustness CSV (Row 2 of figure8_combined_2x3)
df_robust = pd.read_csv(os.path.join(ROBUST_DIR, 'robustness_results.csv'),
                        nrows=100)

# MC Dropout outputs on Train / Test / Val sets
# Scatter and band stats are ALL based on MC mean — consistent with the deployed prediction.
mc_train  = np.load(os.path.join(HERE, 'mc_preds_train.npy'))      # (T, N_train, 3)
y_train_all = np.load(os.path.join(HERE, 'train_labels.npy'))     # (N_train, 3)
mc_test   = np.load(os.path.join(HERE, 'mc_preds_test.npy'))       # (T, N_test, 3)
y_test_all = np.load(os.path.join(HERE, 'test_labels.npy'))       # (N_test, 3)
mc_val    = np.load(os.path.join(HERE, 'mc_preds_val.npy'))        # (T, N_val, 3)
y_val     = np.load(os.path.join(HERE, 'val_labels.npy'))          # (N_val, 3)

# Predictions = MC Dropout mean (= the official model output at deployment time)
y_train_pred_all = mc_train.mean(0)
y_test_pred_all  = mc_test.mean(0)
mu_val,  sg_val  = mc_val.mean(0),  mc_val.std(0)
mu_test, sg_test = y_test_pred_all, mc_test.std(0)

# sigma-scaling temperature (estimated on validation set, no test leakage)
tau = np.sqrt(np.mean(((y_val - mu_val) ** 2) / (sg_val ** 2 + 1e-12), axis=0))
sg_test_cal = tau[None, :] * sg_test

# Aliases used in panel_combined
y_test_mc = y_test_all


def compute_smooth_band(yt, mu_i, sg_i,
                        n_points=BAND_QUERY_POINTS,
                        window_frac=BAND_WINDOW_FRAC):
    """局部线性回归 (LOWESS 风格) 平滑带:
    - 每个查询点用最近 k 个邻居 + tricube 核加权,拟合局部直线 μ ≈ a + b·(yt - q),
      取截距 a 作为 μ(q),避免在边界处出现"平台"(简单均值的回归偏差)。
    - σ 采用同样核权重的加权 RMS。
    自动覆盖 [y_min, y_max] 完整区间。
    """
    n = len(yt)
    k = max(int(n * window_frac), 20)
    y_query = np.linspace(yt.min(), yt.max(), n_points)
    mu_query = np.empty(n_points)
    sg_query = np.empty(n_points)

    for i, q in enumerate(y_query):
        d = np.abs(yt - q)
        idx = np.argpartition(d, min(k, n - 1))[:k]
        d_loc = d[idx]
        yt_loc = yt[idx]
        mu_loc = mu_i[idx]
        sg_loc = sg_i[idx]

        # Tricube 核权重
        max_d = max(d_loc.max(), 1e-12)
        w = (1 - (d_loc / max_d) ** 3) ** 3
        w = np.maximum(w, 1e-12)

        # 加权最小二乘:μ ≈ a + b·(yt - q),取 a
        x = yt_loc - q
        Sw   = w.sum()
        Swx  = (w * x).sum()
        Swxx = (w * x * x).sum()
        Swy  = (w * mu_loc).sum()
        Swxy = (w * x * mu_loc).sum()
        det = Sw * Swxx - Swx * Swx
        if abs(det) < 1e-12:
            mu_query[i] = Swy / Sw
        else:
            mu_query[i] = (Swxx * Swy - Swx * Swxy) / det  # 截距 a

        # σ:核加权 RMS
        sg_query[i] = np.sqrt((w * sg_loc ** 2).sum() / Sw)

    return y_query, mu_query, sg_query


# =================================================================
# Combined panel: 散点(背景) + 置信带(主体) + binned μ + y=x
# =================================================================
def panel_combined(ax, t, panel_label):
    i = t['idx']
    yt_tr, yp_tr = y_train_all[:, i], y_train_pred_all[:, i]
    yt_te, yp_te = y_test_all[:, i],  y_test_pred_all[:, i]
    yt_mc, mu_mc, sg_mc = y_test_mc[:, i], mu_test[:, i], sg_test_cal[:, i]
    band_x, band_mu, band_sg = compute_smooth_band(yt_mc, mu_mc, sg_mc)
    err = np.abs(mu_mc - yt_mc)
    cov1 = np.mean(err <= 1 * sg_mc) * 100
    cov2 = np.mean(err <= 2 * sg_mc) * 100
    rho, _ = spearmanr(sg_mc, err)

    # Axis range
    vmin = min(yt_tr.min(), yp_tr.min(), yt_te.min(), yp_te.min(),
               (band_mu - 2*band_sg).min())
    vmax = max(yt_tr.max(), yp_tr.max(), yt_te.max(), yp_te.max(),
               (band_mu + 2*band_sg).max())
    pad = 0.06 * (vmax - vmin)
    lo, hi = vmin - pad, vmax + pad

    # --- Layer 1 (bottom): train scatter (背景化) ---
    ax.scatter(yt_tr, yp_tr, c=SCATTER_TRAIN_COLOR, s=SCATTER_TRAIN_SIZE,
               alpha=SCATTER_TRAIN_ALPHA, edgecolors='none',
               label='Train', zorder=1)

    # --- Layer 2: test scatter (略明显但仍背景) ---
    ax.scatter(yt_te, yp_te, c=SCATTER_TEST_COLOR, s=SCATTER_TEST_SIZE,
               alpha=SCATTER_TEST_ALPHA, edgecolors='none',
               label='Test', zorder=2)

    # --- Layer 3: ±2σ band (主体, 全范围覆盖) ---
    ax.fill_between(band_x, band_mu - 2*band_sg, band_mu + 2*band_sg,
                    color=BAND_2SIGMA_COLOR, alpha=BAND_2SIGMA_ALPHA,
                    linewidth=0, label=r'$\mu \pm 2\sigma$ (95%)', zorder=3)
    # --- Layer 4: ±1σ band (主体, 全范围覆盖) ---
    ax.fill_between(band_x, band_mu - band_sg, band_mu + band_sg,
                    color=BAND_1SIGMA_COLOR, alpha=BAND_1SIGMA_ALPHA,
                    linewidth=0, label=r'$\mu \pm \sigma$ (68%)', zorder=4)

    # --- Layer 5: smooth μ(x) 曲线 (最显眼) ---
    ax.plot(band_x, band_mu, '-', color=MU_LINE_COLOR,
            linewidth=MU_LINE_WIDTH, label=r'Predicted $\mu(\mathbf{x})$', zorder=5)

    # --- Layer 6: y=x reference ---
    ax.plot([lo, hi], [lo, hi], '--', color='red', linewidth=1.3,
            alpha=0.9, label='y = x', zorder=6)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel(f"True {t['symbol']}{t['unit']}")
    ax.set_ylabel(f"Predicted {t['symbol']}{t['unit']}")
    ax.set_title(f"{panel_label}  {t['name']}", fontweight='bold', pad=8)

    # Stats
    r2_tr = r2_score(yt_tr, yp_tr); r2_te = r2_score(yt_te, yp_te)
    nrmse_tr = np.sqrt(mean_squared_error(yt_tr, yp_tr)) / (yt_tr.max() - yt_tr.min())
    nrmse_te = np.sqrt(mean_squared_error(yt_te, yp_te)) / (yt_te.max() - yt_te.min())
    mae_tr = mean_absolute_error(yt_tr, yp_tr); mae_te = mean_absolute_error(yt_te, yp_te)

    stats = (
        f"$R^2$:    Train={r2_tr:.3f}  Test={r2_te:.3f}\n"
        f"nRMSE: Train={nrmse_tr:.3f}  Test={nrmse_te:.3f}\n"
        f"MAE:   Train={mae_tr:.4f}  Test={mae_te:.4f}\n"
        f"───────────────────────\n"
        f"$\\tau$ = {tau[i]:.3f}   $\\rho(\\sigma,|e|)$ = {rho:.2f}\n"
        f"Coverage@1$\\sigma$ = {cov1:.1f}%\n"
        f"Coverage@2$\\sigma$ = {cov2:.1f}%"
    )
    ax.text(0.97, 0.04, stats, transform=ax.transAxes,
            fontsize=FONT_STATS_BOX, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#bdc3c7',
                      alpha=0.95, linewidth=0.7))

    leg = ax.legend(loc='upper left', frameon=True, edgecolor='#bdc3c7',
                    framealpha=0.92, fontsize=FONT_LEGEND,
                    handlelength=1.8, borderpad=0.4, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.7)


def panel_robustness(ax, t, panel_label, cax):
    r2 = df_robust[t['r2_col']].values
    nrmse = df_robust[t['nrmse_col']].values
    mae = df_robust[t['mae_col']].values
    cmap = mpl.colormaps['RdYlGn_r']
    norm = mpl.colors.Normalize(vmin=mae.min(), vmax=mae.max())

    sc = ax.scatter(r2, nrmse, c=mae, cmap=cmap, norm=norm,
                    s=60, edgecolors='k', linewidths=0.4, alpha=0.9, zorder=3)
    r2_pad    = (r2.max() - r2.min()) * 0.15
    nrmse_pad = (nrmse.max() - nrmse.min()) * 0.15
    ax.set_xlim(r2.min() - r2_pad, r2.max() + r2_pad)
    ax.set_ylim(nrmse.min() - nrmse_pad, nrmse.max() + nrmse_pad)

    mr2 = r2.mean(); mnr = nrmse.mean()
    ax.axvline(mr2, color='#34495e', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)
    ax.axhline(mnr, color='#34495e', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)
    ax.annotate(f'Mean $R^2$={mr2:.4f}',
                xy=(mr2, 0.83), xycoords=('data', 'axes fraction'),
                fontsize=FONT_MEAN_ANNOT, color='#34495e', va='top', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))
    ax.annotate(f'Mean nRMSE={mnr:.4f}',
                xy=(0.97, mnr), xycoords=('axes fraction', 'data'),
                fontsize=FONT_MEAN_ANNOT, color='#34495e', va='center', ha='right',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

    ax.set_xlabel(r'$R^2$')
    ax.set_ylabel('nRMSE')
    ax.set_title(f"{panel_label}  {t['name']}", fontweight='bold', pad=8)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)

    cbar = ax.figure.colorbar(sc, cax=cax)
    cbar.set_label(t['cbar_label'], fontsize=FONT_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)
    cbar.outline.set_linewidth(0.6)


# =================================================================
