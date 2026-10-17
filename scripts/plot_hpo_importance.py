"""
HPO 超参数重要性条形图 — SCI 论文风格
=======================================
从 Optuna 数据库读取超参数重要性, 为每个模型绘制水平条形图.
参数名使用 LaTeX 缩写, 与论文表述一致.

输出:
  models/{model}/figures/hpo_importance.png  — 单模型图
  figures/hpo_importance_all.png             — 多模型合并图

Usage:
    python scripts/plot_hpo_importance.py                # 全部 HPO 模型
    python scripts/plot_hpo_importance.py --model ar     # 单模型
    python scripts/plot_hpo_importance.py --model fm_cfg ddpm  # 多模型
"""

import os, sys, argparse, json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIG_DIR = os.path.join(ROOT_DIR, "figures")

# ═══════════════════════════════════════════════════════════════════
#  绘图配置 — 参考 TeM 论文风格
# ═══════════════════════════════════════════════════════════════════

FONT_SIZE_BASE    = 26
FONT_SIZE_LABEL   = 24
FONT_SIZE_TITLE   = 22
FONT_SIZE_TICK    = 22
FONT_SIZE_ANNO    = 20             # 条形末端数值
FONT_SIZE_LEGEND  = 20

BAR_HEIGHT        = 0.6
BAR_EDGE_COLOR    = "black"
BAR_EDGE_WIDTH    = 1.5
COLOR_TOP         = "#E74C3C"      # 最重要参数
COLOR_OTHER       = "#3498DB"      # 其余参数

GRID_ALPHA        = 0.5
GRID_STYLE        = "--"
GRID_LW           = 1.5
SPINE_LW          = 1.5
TICK_WIDTH        = 1.5

SAVE_DPI          = 600

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "font.size":        FONT_SIZE_BASE,
    "axes.labelsize":   FONT_SIZE_LABEL,
    "axes.titlesize":   FONT_SIZE_TITLE,
    "xtick.labelsize":  FONT_SIZE_TICK,
    "ytick.labelsize":  FONT_SIZE_TICK,
    "axes.linewidth":   SPINE_LW,
    "mathtext.fontset": "stix",
    "savefig.dpi":      SAVE_DPI,
    "savefig.bbox":     "tight",
})

# ═══════════════════════════════════════════════════════════════════
#  模型配置
# ═══════════════════════════════════════════════════════════════════

# 有 HPO 数据库的模型
ALL_HPO_MODELS = ["fm_cfg", "ddpm", "ar", "cinn", "cvae", "cgan", "pa_fm_cfg"]

DISPLAY = {
    "fm_cfg": "cFM-CFG", "ddpm": "cDDPM", "ar": "cAR", "cinn": "cINN",
    "cvae": "cVAE", "cgan": "cWGAN-GP", "pa_fm_cfg": "PA-FM-CFG",
}

# db 内部 study name 映射 (fm_cfg 的 db 里 study 名是 hpo_flow_matching)
STUDY_NAMES = {
    "fm_cfg": "hpo_flow_matching",
}

# ═══════════════════════════════════════════════════════════════════
#  超参数缩写映射 — 论文 LaTeX 表述
# ═══════════════════════════════════════════════════════════════════

PARAM_DISPLAY = {
    # 通用
    "lr":              r"$\eta$",
    "dropout":         r"$\gamma$",
    "hidden_dim":      r"$d_h$",
    "n_blocks":        r"$N_{blk}$",

    # DDPM / DDPM-CFG
    "timesteps":       r"$T$",

    # FM-CFG / DDPM-CFG (CFG 相关)
    "p_uncond":        r"$p_{uc}$",
    "guidance_scale":  r"$w$",
    "sampling_steps":  r"$N_{step}$",

    # cVAE
    "latent_dim":      r"$d_z$",
    "n_layers":        r"$N_{layer}$",
    "beta":            r"$\beta$",

    # cGAN
    "noise_dim":       r"$d_{noise}$",
    "n_g_blocks":      r"$N_G$",
    "n_d_blocks":      r"$N_D$",
    "n_critic":        r"$n_{critic}$",
    "lambda_gp":       r"$\lambda_{GP}$",
    "lr_g":            r"$\eta_G$",
    "lr_d":            r"$\eta_D$",

    # AR
    "dim":             r"$d_{model}$",
    "n_heads":         r"$H$",
    "n_layers":        r"$N_{layer}$",

    # cINN
    "n_flows":         r"$N_{flow}$",

    # PA-FM-CFG
    "lambda_perf":     r"$\lambda_p$",
    "ode_steps":       r"$K_{ode}$",
    "t_min":           r"$t_{min}$",
    "warmup_epochs":   r"$E_{warm}$",
    "ramp_epochs":     r"$E_{ramp}$",
}


def get_display_name(param):
    """获取参数的 LaTeX 缩写, 未知参数保持原名"""
    return PARAM_DISPLAY.get(param, param)


# ═══════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════

def plot_importance_single(model_name, importances):
    """单模型超参数重要性条形图"""
    fig_dir = os.path.join(MODELS_DIR, model_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 按重要性升序 (barh 从下到上, 最重要的在最上面)
    params_sorted = sorted(importances.keys(), key=lambda k: importances[k])
    labels = [get_display_name(p) for p in params_sorted]
    values = [importances[p] for p in params_sorted]

    max_val = max(values)
    colors = [COLOR_TOP if v == max_val else COLOR_OTHER for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.barh(labels, values, color=colors, edgecolor=BAR_EDGE_COLOR,
                   lw=BAR_EDGE_WIDTH, height=BAR_HEIGHT)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=FONT_SIZE_ANNO)

    ax.set_xlabel("Importance")
    ax.set_xlim(0, max_val * 1.25)
    ax.grid(axis="x", linestyle=GRID_STYLE, linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=TICK_WIDTH, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"hpo_importance.{fmt}"), dpi=SAVE_DPI)
    plt.close(fig)
    print(f"    Saved hpo_importance.png/pdf -> {fig_dir}")


def plot_importance_all(all_importances):
    """多模型合并图 (子图网格)"""
    models = list(all_importances.keys())
    n = len(models)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        importances = all_importances[model_name]
        params_sorted = sorted(importances.keys(), key=lambda k: importances[k])
        labels = [get_display_name(p) for p in params_sorted]
        values = [importances[p] for p in params_sorted]
        max_val = max(values) if values else 1
        colors = [COLOR_TOP if v == max_val else COLOR_OTHER for v in values]

        bars = ax.barh(labels, values, color=colors, edgecolor=BAR_EDGE_COLOR,
                       lw=1.0, height=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=FONT_SIZE_ANNO - 6)

        ax.set_title(DISPLAY.get(model_name, model_name), fontsize=FONT_SIZE_TITLE)
        ax.set_xlim(0, max_val * 1.3)
        ax.grid(axis="x", linestyle=GRID_STYLE, linewidth=1.0, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", width=1.0, direction="in", labelsize=FONT_SIZE_TICK - 6)

    # 隐藏多余子图
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"hpo_importance_all.{fmt}"), dpi=SAVE_DPI)
    plt.close(fig)
    print(f"\n  Saved hpo_importance_all.png/pdf -> {FIG_DIR}")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPO 超参数重要性条形图 (SCI 论文风格)")
    parser.add_argument("--model", nargs="*", default=None,
                        help="模型名称 (可多个), 不指定则运行全部 HPO 模型")
    args = parser.parse_args()

    models_to_run = args.model if args.model else ALL_HPO_MODELS

    print("=" * 60)
    print("  HPO Hyperparameter Importance Plots")
    print(f"  Models: {models_to_run}")
    print("=" * 60)

    all_importances = {}

    for m in models_to_run:
        db_path = os.path.join(MODELS_DIR, m, "results", f"optuna_hpo_{m}.db")
        if not os.path.exists(db_path):
            print(f"\n  [{m}] No HPO database, skip")
            continue

        study_name = STUDY_NAMES.get(m, f"hpo_{m}")
        storage = f"sqlite:///{db_path}"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            print(f"\n  [{m}] Study '{study_name}' not found in DB, skip")
            continue

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            print(f"\n  [{m}] Only {len(completed)} completed trials, skip")
            continue

        print(f"\n  [{DISPLAY.get(m, m)}] {len(completed)} trials, best={study.best_value:.6f}")

        try:
            evaluator = FanovaImportanceEvaluator(seed=42)
            importances = get_param_importances(study, evaluator=evaluator)
            all_importances[m] = dict(importances)
            plot_importance_single(m, importances)
        except Exception as e:
            print(f"    Failed: {e}")

    # 合并图
    if len(all_importances) > 1:
        print("\n  Plotting combined figure...")
        plot_importance_all(all_importances)

    # 保存重要性数据 (json)
    if all_importances:
        json_data = {}
        for m, imp in all_importances.items():
            json_data[m] = {
                "display": DISPLAY.get(m, m),
                "importances": {k: {"raw_name": k, "display": get_display_name(k), "value": v}
                                for k, v in imp.items()},
            }
        os.makedirs(FIG_DIR, exist_ok=True)
        with open(os.path.join(FIG_DIR, "hpo_importance.json"), "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  Saved hpo_importance.json -> {FIG_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
