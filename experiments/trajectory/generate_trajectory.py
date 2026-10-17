"""
FM-CFG 生成轨迹可视化 (Nature-style)
=====================================
给定目标条件, 从噪声 x_0 ~ N(0, I) 通过 100 步 ODE 积分到 x_1, 记录全过程,
可视化几何参数、性能预测、PCA 流形如何演化.

目标 (denorm): m=15.25 kg/s, eta=0.88, pi=1.62
轨迹数: 10 (不同种子, 同目标 → 体现多样性收敛)

输出:
  results/trajectory_data.npz       轨迹原始数据 (trajectory, velocity, vit_pred)
  figures/trajectory_main.png       Nature 风格 2x2 主图
"""

import os, sys, json
import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from shared.data_utils import load_norm_stats, normalize, denormalize
from shared.evaluation import PerformanceEvaluator
from models.fm_cfg.model import ConditionalFlowMatching
from models.ddpm.model import ConditionalDDPM

RES_DIR = os.path.join(EXP_DIR, "results")
FIG_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  实验配置
# ════════════════════════════════════════════════════════════════
TARGET_DE     = np.array([15.371,0.8841,1.6274])    # 目标 (denorm): m, eta, pi
N_TRAJ        = 10                                # 轨迹数 (不同种子)
N_STEPS       = 100                               # ODE 步数 (与 HPO best 一致)
GUIDANCE_SCALE = 1.964                            # FM-CFG HPO best

FM_CFG_HPO_JSON = os.path.join(ROOT_DIR, "models", "fm_cfg", "results", "hpo_best_fm_cfg.json")
FM_CFG_CKPT     = os.path.join(ROOT_DIR, "models", "fm_cfg", "checkpoints", "fm_cfg_hpo.pth")
DDPM_HPO_JSON   = os.path.join(ROOT_DIR, "models", "ddpm", "results", "hpo_best_ddpm.json")
DDPM_CKPT       = os.path.join(ROOT_DIR, "models", "ddpm", "checkpoints", "ddpm_hpo.pth")


# ════════════════════════════════════════════════════════════════
#  全局字体配置 (SCI 论文风格)
#  - 集中放在此处, 修改一处即可统一影响组合主图与三张独立子图
#  - 与 scripts/plot_regression.py 风格保持一致
# ════════════════════════════════════════════════════════════════
# 字族
FONT_FAMILY          = "serif"
FONT_SERIF           = ["Times New Roman"]
MATHTEXT_FONTSET     = "stix"

# 基础字号 (rcParams 全局, 影响所有未显式指定的文本)
FONT_SIZE_BASE       = 18

# 坐标轴
FONT_SIZE_AXIS_LABEL = 22    # x / y 轴标签 (例: "PC1 (13.1%)")
FONT_SIZE_TICK       = 18    # 主刻度数字

# 标题
FONT_SIZE_TITLE      = 24    # panel 子标题 (例: "a    FM-CFG (ODE)")

# 图例
FONT_SIZE_LEGEND     = 17    # 图例文字 (起点/终点/带宽等)

# 注释 (inline 文本)
FONT_SIZE_ANNOTATION = 16    # 速度图里 "High initial drift" / "Fine-tuning" 等

# Colorbar
FONT_SIZE_CBAR_LABEL = 18    # colorbar 轴标签 ("Generation time τ")
FONT_SIZE_CBAR_TICK  = 15    # colorbar 刻度数字

# 间距
TITLE_PAD            = 11    # 标题与轴的纵向间距
AXIS_LABEL_PAD       = 7     # 轴标签与刻度的间距

# 框 / 刻度宽度
SPINE_LW             = 1.5   # 四边框线宽
TICK_LEN_MAJOR       = 6
TICK_LEN_MINOR       = 3
TICK_WIDTH_MAJOR     = 1.1
TICK_WIDTH_MINOR     = 0.85

# 输出
SAVE_DPI             = 600
SAVE_PAD             = 0.22


# ════════════════════════════════════════════════════════════════
#  自定义采样: 记录全部 ODE 轨迹
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def sample_with_trajectory(fm, target_norm, n_traj, n_steps, gs, seeds, device):
    """
    返回:
      traj:  (n_steps+1, n_traj, 21) numpy
      vel:   (n_steps,   n_traj, 21) numpy  每步速度 v(x_t, t, c) 含 CFG 引导
      ts:    (n_steps+1,) numpy
    """
    target_tensor = torch.tensor(target_norm, dtype=torch.float32, device=device)
    target_batch  = target_tensor.unsqueeze(0).repeat(n_traj, 1)
    null_cond     = torch.zeros_like(target_batch)

    dt = 1.0 / n_steps

    # 不同种子的初始噪声 (CPU 上生成, 然后搬到 device 保证可复现)
    x_init = np.stack([
        np.random.RandomState(s).randn(fm.geom_dim).astype(np.float32)
        for s in seeds
    ])
    x = torch.tensor(x_init, device=device)

    traj = [x.cpu().numpy().copy()]
    vel  = []
    for i in range(n_steps):
        t = torch.full((n_traj,), i * dt, device=device)
        v_cond   = fm.net(x, t, target_batch)
        v_uncond = fm.net(x, t, null_cond)
        v = v_uncond + gs * (v_cond - v_uncond)
        vel.append(v.cpu().numpy().copy())
        x = x + v * dt
        traj.append(x.cpu().numpy().copy())

    return np.array(traj), np.array(vel), np.linspace(0, 1, n_steps + 1)


# ════════════════════════════════════════════════════════════════
#  ViT 评估中间状态
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def vit_eval_trajectory(vit, traj, device):
    """对每一步 (clip 到 [0,1]) 做 ViT 预测.
    traj: (T, N, 21) → return (T, N, 3) 归一化空间.
    """
    T, N, D = traj.shape
    x = torch.tensor(traj.reshape(T * N, D), dtype=torch.float32, device=device)
    x_clipped = x.clamp(0, 1).unsqueeze(2)        # (T*N, 21, 1)
    pred = vit(x_clipped)
    return pred.cpu().numpy().reshape(T, N, 3)


def load_fm_cfg(device):
    with open(FM_CFG_HPO_JSON) as f:
        params = json.load(f)["best_params"]
    model = ConditionalFlowMatching(
        geom_dim=21, cond_dim=3,
        hidden_dim=params["hidden_dim"], n_blocks=params["n_blocks"],
        p_uncond=params["p_uncond"], guidance_scale=params["guidance_scale"],
        sampling_steps=params["sampling_steps"], dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(FM_CFG_CKPT, map_location=device))
    model.eval()
    return model, params


def load_ddpm(device):
    with open(DDPM_HPO_JSON) as f:
        params = json.load(f)["best_params"]
    model = ConditionalDDPM(
        geom_dim=21, cond_dim=3,
        timesteps=params["timesteps"],
        hidden_dim=params["hidden_dim"], n_blocks=params["n_blocks"],
        dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(DDPM_CKPT, map_location=device))
    model.eval()
    return model, params


@torch.no_grad()
def sample_ddpm_with_trajectory(ddpm, target_norm, n_traj, seeds, device):
    """cDDPM 采样, 记录 reverse-process 全程轨迹.
    返回:
      traj:  (T+1, n_traj, 21) 倒序对齐到 t∈[0,1]: 起点=噪声(t=0), 终点=设计(t=1)
      step:  (T,   n_traj)     每步位移 ||x_{i-1} - x_i||
      ts:    (T+1,)             linspace(0, 1)
    """
    T = ddpm.timesteps
    cond = torch.tensor(target_norm, dtype=torch.float32, device=device)
    cond_batch = cond.unsqueeze(0).repeat(n_traj, 1)

    # 起始噪声 (与 FM 用同种子, 保证可对比)
    x_init = np.stack([
        np.random.RandomState(s).randn(ddpm.geom_dim).astype(np.float32)
        for s in seeds
    ])
    x = torch.tensor(x_init, device=device)

    traj_rev = [x.cpu().numpy().copy()]
    step_rev = []
    for i in reversed(range(T)):
        t_idx = torch.full((n_traj,), i, device=device, dtype=torch.long)
        eps = ddpm.net(x, t_idx, cond_batch)
        alpha     = ddpm.alphas[i]
        alpha_bar = ddpm.alphas_cumprod[i]
        beta      = ddpm.betas[i]
        mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * eps)
        if i > 0:
            x_new = mean + beta.sqrt() * torch.randn_like(x)
        else:
            x_new = mean
        step_rev.append((x_new - x).norm(dim=-1).cpu().numpy().copy())
        x = x_new
        traj_rev.append(x.cpu().numpy().copy())

    traj = np.array(traj_rev)            # (T+1, n_traj, 21)
    step = np.array(step_rev)            # (T,   n_traj)
    ts   = np.linspace(0, 1, T + 1)
    return traj, step, ts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  FM-CFG Generation Trajectory Visualization")
    print(f"  Device: {device}")
    print(f"  Target (denorm): m={TARGET_DE[0]}, eta={TARGET_DE[1]}, pi={TARGET_DE[2]}")
    print(f"  N_TRAJ={N_TRAJ}, N_STEPS={N_STEPS}, GS={GUIDANCE_SCALE}")
    print("=" * 60)

    # ── 加载 ──
    fm, fm_params = load_fm_cfg(device)
    print(f"  FM-CFG params: {fm_params}")
    ddpm, ddpm_params = load_ddpm(device)
    print(f"  cDDPM params:  {ddpm_params}")

    stats = load_norm_stats()
    target_norm = normalize(TARGET_DE, stats["label_min"], stats["label_max"])
    print(f"  Target (norm):   {target_norm.round(4).tolist()}")

    # ── 采样 ──
    seeds = list(range(42, 42 + N_TRAJ))
    print(f"\n  Sampling FM-CFG ({N_STEPS} steps)...")
    torch.manual_seed(42)                              # 保证 torch RNG 可复现
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    fm_traj, fm_vel, fm_ts = sample_with_trajectory(
        fm, target_norm, N_TRAJ, N_STEPS, GUIDANCE_SCALE, seeds, device)
    fm_step_mag = np.linalg.norm(fm_vel, axis=-1) * (1.0 / N_STEPS)
    print(f"    FM trajectory: {fm_traj.shape}")

    print(f"  Sampling cDDPM ({ddpm.timesteps} steps)...")
    torch.manual_seed(42)                              # cDDPM 中间噪声注入也可复现
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    dd_traj, dd_step, dd_ts = sample_ddpm_with_trajectory(
        ddpm, target_norm, N_TRAJ, seeds, device)
    print(f"    cDDPM trajectory: {dd_traj.shape}")

    # ── 联合 PCA (两模型共享同一坐标系, 便于左右对比) ──
    all_states = np.vstack([fm_traj.reshape(-1, 21), dd_traj.reshape(-1, 21)])
    pca = PCA(n_components=2)
    pca.fit(all_states)
    fm_2d = pca.transform(fm_traj.reshape(-1, 21)).reshape(fm_traj.shape[0], fm_traj.shape[1], 2)
    dd_2d = pca.transform(dd_traj.reshape(-1, 21)).reshape(dd_traj.shape[0], dd_traj.shape[1], 2)
    print(f"  Joint PCA explained variance: "
          f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    # ── 保存 ──
    np.savez(os.path.join(RES_DIR, "trajectory_data.npz"),
             fm_traj=fm_traj, fm_vel=fm_vel, fm_step_mag=fm_step_mag,
             fm_ts=fm_ts, fm_2d=fm_2d,
             dd_traj=dd_traj, dd_step=dd_step, dd_ts=dd_ts, dd_2d=dd_2d,
             target_de=TARGET_DE, target_norm=target_norm,
             pca_components=pca.components_,
             pca_var_ratio=pca.explained_variance_ratio_,
             seeds=seeds)

    # ════════════════════════════════════════════════════════════
    #  rcParams — 应用文件顶部集中定义的字号 / 风格常量
    # ════════════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.family":         FONT_FAMILY,
        "font.serif":          FONT_SERIF,
        "mathtext.fontset":    MATHTEXT_FONTSET,
        "font.size":           FONT_SIZE_BASE,
        "axes.linewidth":      SPINE_LW,
        "axes.labelsize":      FONT_SIZE_AXIS_LABEL,
        "axes.titlesize":      FONT_SIZE_TITLE,
        "axes.titleweight":    "bold",
        "axes.labelpad":       AXIS_LABEL_PAD,
        "xtick.labelsize":     FONT_SIZE_TICK, "ytick.labelsize": FONT_SIZE_TICK,
        "xtick.direction":     "in", "ytick.direction": "in",
        "xtick.major.size":    TICK_LEN_MAJOR, "ytick.major.size": TICK_LEN_MAJOR,
        "xtick.major.width":   TICK_WIDTH_MAJOR, "ytick.major.width": TICK_WIDTH_MAJOR,
        "xtick.minor.size":    TICK_LEN_MINOR, "ytick.minor.size": TICK_LEN_MINOR,
        "xtick.minor.width":   TICK_WIDTH_MINOR, "ytick.minor.width": TICK_WIDTH_MINOR,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top":           True, "ytick.right": True,
        "legend.fontsize":     FONT_SIZE_LEGEND,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "#bdc3c7",
        "legend.fancybox":     False,
        "legend.handlelength": 1.6,
        "savefig.dpi":         SAVE_DPI, "savefig.bbox": "tight",
        "savefig.pad_inches":  SAVE_PAD,
    })

    CMAP_TIME    = "viridis"
    cmap = plt.get_cmap(CMAP_TIME)
    END_COLOR    = cmap(0.97)
    START_EDGE   = "#1A1A1A"
    MEAN_COLOR   = "#1A1A1A"
    TRAJ_COLOR   = "#9AA5B3"

    # 联合 PCA 坐标轴范围 (上下/左右各留 2% 余量, 减少子图内白边)
    all_2d = np.concatenate([fm_2d.reshape(-1, 2), dd_2d.reshape(-1, 2)], axis=0)
    pc_xmin, pc_xmax = all_2d[:, 0].min(), all_2d[:, 0].max()
    pc_ymin, pc_ymax = all_2d[:, 1].min(), all_2d[:, 1].max()
    mg_x = (pc_xmax - pc_xmin) * 0.025
    mg_y = (pc_ymax - pc_ymin) * 0.025

    vel_mag_arr = np.linalg.norm(fm_vel, axis=-1)
    vel_t   = fm_ts[:-1]
    vel_mean = vel_mag_arr.mean(axis=1)
    vel_std  = vel_mag_arr.std(axis=1)

    # ════════════════════════════════════════════════════════════
    #  绘制函数 (combined / standalone 通用)
    # ════════════════════════════════════════════════════════════
    def draw_pca(ax, traj_2d, ts_arr, title_left, sample_steps,
                 show_ylabel=True):
        norm_t = Normalize(0, 1)
        for n in range(N_TRAJ):
            pts = traj_2d[:, n, :]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm_t,
                                linewidths=1.6, alpha=0.80,
                                capstyle="round", zorder=2)
            lc.set_array(ts_arr[:-1])
            ax.add_collection(lc)
        # 终点 (设计) — 与 viridis 末端色一致, 黑细边
        ax.scatter(traj_2d[-1, :, 0], traj_2d[-1, :, 1],
                   s=88, facecolors=END_COLOR, edgecolors=START_EDGE,
                   linewidths=1.1, marker="o", zorder=6,
                   label=r"$x_1$  (generated design)")
        # 起点 (噪声) — zorder 提高, 不被密集轨迹遮挡
        ax.scatter(traj_2d[0, :, 0], traj_2d[0, :, 1],
                   s=72, facecolors="white", edgecolors=START_EDGE,
                   linewidths=1.3, zorder=8,
                   label=r"$x_0 \sim \mathcal{N}(0, I)$")
        ax.set_xlim(pc_xmin - mg_x, pc_xmax + mg_x)
        ax.set_ylim(pc_ymin - mg_y, pc_ymax + mg_y)
        # PC1 / PC2 不带百分比 (变量名简洁, 便于跨图对照)
        ax.set_xlabel("PC1")
        if show_ylabel:
            ax.set_ylabel("PC2")
        # 标题居中显示 (a)(b)(c) 放在子图中间
        ax.set_title(title_left, loc="center", pad=TITLE_PAD)
        # 右下角 inline 注释 (采样步数)
        ax.text(0.97, 0.03, f"{sample_steps} steps",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONT_SIZE_ANNOTATION, color="#555", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#bdc3c7", lw=0.7, alpha=0.92))
        # 图例 (起点/终点标记)
        leg = ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND,
                        handletextpad=0.5, labelspacing=0.55,
                        borderpad=0.45)
        leg.set_zorder(20)

    def draw_velocity(ax, title_left):
        for n in range(N_TRAJ):
            ax.plot(vel_t, vel_mag_arr[:, n], color=TRAJ_COLOR,
                    lw=0.9, alpha=0.55, zorder=2)
        ax.fill_between(vel_t, vel_mean - vel_std, vel_mean + vel_std,
                        color=TRAJ_COLOR, alpha=0.25, zorder=1,
                        linewidth=0, label=r"$\pm 1\sigma$ band")
        ax.plot(vel_t, vel_mean, color=MEAN_COLOR, lw=2.6, zorder=4,
                label="Mean over 10 trajectories")
        # inline 注释: 早期高速漂移阶段
        ax.annotate("High initial drift\n(noise → manifold)",
                    xy=(0.04, vel_mean[2]), xytext=(0.20, vel_mean[2] * 1.05),
                    fontsize=FONT_SIZE_ANNOTATION, color="#444", fontstyle="italic",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#666",
                                    lw=1.0, alpha=0.7,
                                    connectionstyle="arc3,rad=0.15"))
        # inline 注释: 后期精修阶段
        ax.annotate("Fine-tuning\nphase",
                    xy=(0.92, vel_mean[-3]),
                    xytext=(0.65, vel_mean[-3] * 1.18),
                    fontsize=FONT_SIZE_ANNOTATION, color="#444", fontstyle="italic",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#666",
                                    lw=1.0, alpha=0.7,
                                    connectionstyle="arc3,rad=-0.15"))
        ax.set_xlim(0, 1.0)
        # 全图统一用 "Generation time t" (与 (a)(b) colorbar 同一变量)
        # 速度场写作 v_θ(x_t, t, y): θ=网络参数, x_t=t 时刻状态, y=条件向量
        ax.set_xlabel(r"Generation time  $t$")
        ax.set_ylabel(r"Velocity magnitude  $\|v_\theta(x_t,\, t,\, y)\|_2$")
        # 标题居中显示 (a)(b)(c) 放在子图中间
        ax.set_title(title_left, loc="center", pad=TITLE_PAD)
        ax.grid(True, ls=":", lw=0.55, alpha=0.45, axis="y", zorder=0)
        ax.set_axisbelow(True)
        # 图例 (±1σ 带 + 均值)
        leg = ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND,
                        handletextpad=0.6, labelspacing=0.6,
                        borderpad=0.45)
        leg.set_zorder(20)

    # ════════════════════════════════════════════════════════════
    #  组合主图: 上 (a+b) 与下 (c) 等宽对齐
    #  策略: 水平 colorbar 放最顶部, 让 (a)(b)(c) 共享相同 left/right 边距
    # ════════════════════════════════════════════════════════════
    LEFT, RIGHT = 0.065, 0.985
    fig = plt.figure(figsize=(13.0, 9.6))
    # 顶部留出 colorbar 空间, 压缩左右白边
    # hspace 留足以避免 (c) 标题撞 (a)(b) 的 x 轴标签
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85],
                          hspace=0.34, wspace=0.14,
                          left=LEFT, right=RIGHT,
                          top=0.91, bottom=0.085)
    ax_fm = fig.add_subplot(gs[0, 0])
    ax_dd = fig.add_subplot(gs[0, 1])
    ax_v  = fig.add_subplot(gs[1, :])    # 横跨两列, 与 (a)+(b) 总宽度对齐

    draw_pca(ax_fm, fm_2d, fm_ts,
             "(a) cFM-CFG  (ODE)", sample_steps=N_STEPS)
    draw_pca(ax_dd, dd_2d, dd_ts,
             "(b) cDDPM  (SDE)", sample_steps=ddpm.timesteps,
             show_ylabel=False)

    # 水平 colorbar — 顶部居中
    # 字号采用顶部 FONT_SIZE_CBAR_LABEL / FONT_SIZE_CBAR_TICK 统一控制
    # 标签放在 colorbar 下方 (居于 colorbar 与 (a)(b) 之间, 即 "图中间")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cbar_ax = fig.add_axes([0.32, 0.955, 0.38, 0.020])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Generation time  $t$",
                   fontsize=FONT_SIZE_CBAR_LABEL, labelpad=5)
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK, length=4, width=0.9)
    cbar.outline.set_linewidth(0.9)
    cbar_ax.xaxis.set_ticks_position("top")        # 刻度放上方
    cbar_ax.xaxis.set_label_position("bottom")     # 标签放下方 (居中显示)

    draw_velocity(ax_v, "(c) cFM-CFG  ODE velocity field magnitude")

    out_main = os.path.join(FIG_DIR, "trajectory_main.png")
    fig.savefig(out_main)
    plt.close(fig)
    print(f"\n  Saved combined: {out_main}")

    # ════════════════════════════════════════════════════════════
    #  单独保存 3 个子图
    # ════════════════════════════════════════════════════════════
    def _save_pca_standalone(traj_2d, ts_arr, title, sample_steps, fname):
        # 独立 PCA 子图: 保留 draw_pca 内设的联合 PCA 范围
        # 这样 cFM-CFG / cDDPM 两张独立子图坐标刻度严格一致, 便于直接对比
        fig_s, ax_s = plt.subplots(figsize=(7.5, 5.0))
        fig_s.subplots_adjust(left=0.085, right=0.88, top=0.91, bottom=0.13)
        draw_pca(ax_s, traj_2d, ts_arr, title, sample_steps)
        # 不覆盖 xlim/ylim, 使用 draw_pca 内的联合范围 (pc_xmin/xmax/ymin/ymax)

        sm_s = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
        cb = fig_s.colorbar(sm_s, ax=ax_s, pad=0.015, aspect=22, shrink=0.95)
        cb.set_label(r"Generation time  $t$",
                     fontsize=FONT_SIZE_CBAR_LABEL, labelpad=5)
        cb.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK)
        cb.outline.set_linewidth(0.9)
        out = os.path.join(FIG_DIR, fname)
        fig_s.savefig(out)
        plt.close(fig_s)
        print(f"  Saved standalone: {out}")

    _save_pca_standalone(fm_2d, fm_ts,
                         "cFM-CFG  (ODE)", N_STEPS,
                         "panel_a_fm_pca.png")
    _save_pca_standalone(dd_2d, dd_ts,
                         "cDDPM  (SDE)", ddpm.timesteps,
                         "panel_b_ddpm_pca.png")

    # 独立速度子图: 纵横比 W:H ≈ 3.35, 边距大幅收紧消除空白
    fig_v, ax_vs = plt.subplots(figsize=(13.0, 4.2))
    fig_v.subplots_adjust(left=0.055, right=0.992, top=0.91, bottom=0.165)
    draw_velocity(ax_vs, "cFM-CFG  ODE velocity field magnitude")
    out_v = os.path.join(FIG_DIR, "panel_c_velocity.png")
    fig_v.savefig(out_v)
    plt.close(fig_v)
    print(f"  Saved standalone: {out_v}")

    print("\nDone!")


if __name__ == "__main__":
    main()
