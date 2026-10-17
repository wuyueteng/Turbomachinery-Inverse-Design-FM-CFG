"""
FM 消融模型推理脚本 — 硬编码配置, 无需 HPO 结果
================================================
直接加载 FM-CFG 最佳架构配置 + FM 训练权重, 生成样本并保存.
输出格式与 scripts/inference.py 完全一致, 供绘图脚本直接读取.

Usage:
    python models/fm/infer.py
    python models/fm/infer.py --gpu 0 --n_cond 200 --n_samples 20
"""

import os, sys, json, csv, argparse
import numpy as np
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(os.path.dirname(MODEL_DIR))
sys.path.insert(0, ROOT_DIR)

from shared.data_utils import get_dataloaders, denormalize, GEOM_NAMES, PERF_NAMES
from shared.evaluation import PerformanceEvaluator
from models.fm.model import ConditionalFM

# ═══════════════════════════════════════════════════════════════════
#  模型配置 — 直接复用 FM-CFG HPO 最佳超参 (去掉 CFG)
# ═══════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    "hidden_dim":     512,
    "n_blocks":       5,
    "dropout":        0.081,
    "sampling_steps": 100,
}
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoints", "fm.pth")
OUT_DIR   = os.path.join(MODEL_DIR, "results", "inference")
MODEL_NAME = "fm"
DISPLAY_NAME = "FM"


# ═══════════════════════════════════════════════════════════════════
#  保存函数 (与 scripts/inference.py 输出格式一致)
# ═══════════════════════════════════════════════════════════════════

def save_results(result, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pred_de  = result["pred_denorm"]
    targ_de  = result["target_denorm"]
    all_geom = result["all_geom"]
    geom_de  = denormalize(all_geom, stats["geom_min"], stats["geom_max"])

    # npy
    np.save(os.path.join(out_dir, "geometry_norm.npy"), all_geom)
    np.save(os.path.join(out_dir, "geometry_denorm.npy"), geom_de)
    np.save(os.path.join(out_dir, "pred_performance.npy"), pred_de)
    np.save(os.path.join(out_dir, "target_performance.npy"), targ_de)

    # csv
    header = GEOM_NAMES + [f"Target_{n}" for n in PERF_NAMES] + [f"Pred_{n}" for n in PERF_NAMES]
    rows = np.hstack([geom_de, targ_de, pred_de])
    with open(os.path.join(out_dir, "inference_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.6f}" for v in row])

    # metrics
    dim_names = ["flow", "eff", "pr"]
    r2, nrmse, mre, mae = [], [], [], []
    for d in range(3):
        res = pred_de[:, d] - targ_de[:, d]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((targ_de[:, d] - targ_de[:, d].mean()) ** 2) + 1e-12
        r2.append(1 - ss_res / ss_tot)
        rmse = np.sqrt(np.mean(res ** 2))
        tgt_range = targ_de[:, d].max() - targ_de[:, d].min() + 1e-12
        nrmse.append(rmse / tgt_range)
        mre.append(np.mean(np.abs(res) / (np.abs(targ_de[:, d]) + 1e-12)) * 100)
        mae.append(float(np.mean(np.abs(res))))

    # txt
    with open(os.path.join(out_dir, "inference_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {DISPLAY_NAME}\n")
        f.write(f"Samples: {len(pred_de)}\n")
        f.write(f"Config: {MODEL_CONFIG}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<20s} {'Flow':>12s} {'Efficiency':>12s} {'PR':>12s}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'R2':<20s} {r2[0]:>12.4f} {r2[1]:>12.4f} {r2[2]:>12.4f}\n")
        f.write(f"{'nRMSE':<20s} {nrmse[0]:>12.4f} {nrmse[1]:>12.4f} {nrmse[2]:>12.4f}\n")
        f.write(f"{'MRE (%)':<20s} {mre[0]:>12.2f} {mre[1]:>12.2f} {mre[2]:>12.2f}\n")
        f.write(f"{'MAE':<20s} {mae[0]:>12.4f} {mae[1]:>12.4f} {mae[2]:>12.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'MSE (norm)':<20s} {result['mse_mean']:>12.6f}\n")
        f.write(f"{'Hit Rate':<20s} {result['hit_rate']:>12.4f}\n")
        f.write(f"{'Diversity (dist)':<20s} {result['diversity_dist']:>12.4f}\n")
        f.write(f"{'Diversity (std)':<20s} {result['diversity_std']:>12.4f}\n")

    # json
    metrics = {
        "model": MODEL_NAME, "display_name": DISPLAY_NAME,
        "config": MODEL_CONFIG,
        "n_samples": int(len(pred_de)),
        "mse_norm": result["mse_mean"],
        "hit_rate": result["hit_rate"],
        "diversity_dist": result["diversity_dist"],
        "diversity_std": result["diversity_std"],
    }
    for d, dn in enumerate(dim_names):
        metrics[f"r2_{dn}"]    = float(r2[d])
        metrics[f"nrmse_{dn}"] = float(nrmse[d])
        metrics[f"mre_{dn}"]   = float(mre[d])
        metrics[f"mae_{dn}"]   = float(mae[d])
    with open(os.path.join(out_dir, "inference_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved: {out_dir}/  (npy + csv + txt + json)")


# ═══════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FM 消融推理")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_cond", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  FM Ablation Inference")
    print(f"  Device: {device}")
    print(f"  Config: {MODEL_CONFIG}")
    print(f"  Checkpoint: {CKPT_PATH}")
    print("=" * 60)

    # 加载模型
    model = ConditionalFM(geom_dim=21, cond_dim=3, **MODEL_CONFIG).to(device)
    if not os.path.exists(CKPT_PATH):
        print(f"\n  [ERROR] Checkpoint not found: {CKPT_PATH}")
        print(f"  Run 'python models/fm/train.py' first.")
        return
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"  Loaded checkpoint, params: {sum(p.numel() for p in model.parameters()):,}")

    # 数据
    _, _, _, info = get_dataloaders(batch_size=256, use_merged=True)
    stats = info["stats"]
    test_label = info["test_label"]

    rng = np.random.RandomState(42)
    eval_idx = rng.choice(len(test_label), args.n_cond, replace=False)
    conditions = test_label[eval_idx]

    # 评估
    evaluator = PerformanceEvaluator(str(device))

    def sample_fn(c, n):
        return model.sample(c, n_samples=n)

    print(f"\n  Evaluating {args.n_cond} conditions x {args.n_samples} samples...")
    result = evaluator.evaluate_batch(sample_fn, conditions, stats,
                                       n_samples=args.n_samples)

    # 打印摘要
    print(f"\n  MSE={result['mse_mean']:.6f}  HR={result['hit_rate']:.4f}  "
          f"R2=[{result['r2_flow']:.4f},{result['r2_eff']:.4f},{result['r2_pr']:.4f}]")

    # 保存
    save_results(result, stats, OUT_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
