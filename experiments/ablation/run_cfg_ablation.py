"""
CFG 消融实验 — FM vs FM-CFG
============================
读取 FM (无 CFG) 与 FM-CFG (有 CFG) 的 inference 结果, 汇总对比.
评估流程: ViT 性能预测器 → 比较预测性能与目标条件.

前置条件:
  models/fm/results/inference/       (python models/fm/infer.py)
  models/fm_cfg/results/inference/   (python scripts/inference.py --model fm_cfg)

输出:
  experiments/ablation/results/cfg_ablation.{json,txt}

Usage:
    python experiments/ablation/run_cfg_ablation.py
"""

import os, sys, json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR    = os.path.join(ROOT_DIR, "experiments", "ablation", "results")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NO_CFG   = "fm"
MODEL_WITH_CFG = "fm_cfg"
DISPLAY = {"fm": "FM", "fm_cfg": "cFM-CFG"}

# (显示名, JSON 键, 格式, higher_better)
METRICS = [
    ("MSE",        "mse_norm",       "{:.6f}", False),
    ("R2(Flow)",   "r2_flow",        "{:.4f}", True),
    ("R2(Eff)",    "r2_eff",         "{:.4f}", True),
    ("R2(PR)",     "r2_pr",          "{:.4f}", True),
    ("nRMSE(F)",   "nrmse_flow",     "{:.4f}", False),
    ("nRMSE(E)",   "nrmse_eff",      "{:.4f}", False),
    ("nRMSE(P)",   "nrmse_pr",       "{:.4f}", False),
    ("MAE(Flow)",  "mae_flow",       "{:.4f}", False),
    ("MAE(Eff)",   "mae_eff",        "{:.4f}", False),
    ("MAE(PR)",    "mae_pr",         "{:.4f}", False),
    ("Diversity",  "diversity_dist", "{:.4f}", None),
]


def load_metrics(model_name):
    path = os.path.join(MODELS_DIR, model_name, "results", "inference", "inference_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    bar = "=" * 72
    print(bar)
    print("  CFG Ablation: FM vs FM-CFG")
    print(bar)

    m_no  = load_metrics(MODEL_NO_CFG)
    m_yes = load_metrics(MODEL_WITH_CFG)

    if m_no is None:
        print(f"\n  [{MODEL_NO_CFG}] Missing inference. Run 'python models/fm/infer.py' first.")
        return
    if m_yes is None:
        print(f"\n  [{MODEL_WITH_CFG}] Missing inference. Run 'python scripts/inference.py --model fm_cfg' first.")
        return

    print(f"\n  {'Metric':<14s} {DISPLAY[MODEL_NO_CFG]:>12s} {DISPLAY[MODEL_WITH_CFG]:>12s} {'Delta':>14s}")
    print("  " + "-" * 58)

    comparison = {}
    for name, key, fmt, _ in METRICS:
        v_no  = float(m_no.get(key, 0))
        v_yes = float(m_yes.get(key, 0))
        delta = v_yes - v_no
        delta_str = ("+" if delta > 0 else "") + fmt.format(delta)
        print(f"  {name:<14s} {fmt.format(v_no):>12s} {fmt.format(v_yes):>12s} {delta_str:>14s}")
        comparison[key] = {"no_cfg": v_no, "with_cfg": v_yes, "delta": delta}

    result = {
        "no_cfg":   MODEL_NO_CFG,
        "with_cfg": MODEL_WITH_CFG,
        "metrics":  comparison,
    }

    with open(os.path.join(OUT_DIR, "cfg_ablation.json"), "w") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(OUT_DIR, "cfg_ablation.txt"), "w", encoding="utf-8") as f:
        f.write(f"CFG Ablation: {DISPLAY[MODEL_NO_CFG]} vs {DISPLAY[MODEL_WITH_CFG]}\n")
        f.write(f"{'Metric':<16s} {'No CFG':>12s} {'With CFG':>12s} {'Delta':>14s}\n")
        f.write("-" * 60 + "\n")
        for key, vals in comparison.items():
            f.write(f"{key:<16s} {vals['no_cfg']:>12.6f} {vals['with_cfg']:>12.6f} {vals['delta']:>+14.6f}\n")

    print(f"\n  Saved: {OUT_DIR}/cfg_ablation.{{json,txt}}")
    print("\n" + bar)


if __name__ == "__main__":
    main()
