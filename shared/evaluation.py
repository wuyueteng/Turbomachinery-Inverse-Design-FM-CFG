"""
性能评估工具
===================
- 100 个目标条件 × 10 个生成样本，直接评估全部（无 Top-K 筛选）
- 多样性指标：参数标准差、平均成对距离
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "performance_predictor"))
from vit import ViT


class PerformanceEvaluator:
    """ViT 性能预测模型 + 综合评估"""

    PREDICTOR_CKPT = os.path.join(
        ROOT_DIR, "models", "performance_predictor", "checkpoints", "best_model_in_testdataset.pth"
    )

    def __init__(self, device="cuda", checkpoint_path=None):
        self.device = device
        ckpt_path = checkpoint_path or os.environ.get('TURBO_PREDICTOR_CHECKPOINT') or self.PREDICTOR_CKPT
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Missing predictor checkpoint: {ckpt_path}. Train with python scripts/train.py --model performance_predictor, then pass --predictor-run or set TURBO_PREDICTOR_CHECKPOINT to its best.pt.')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        params = dict(lenth=1, channels=21, num_classes=3, dim=256, depth=14,
                      heads=12, mlp_dim=256, dropout=0.085, emb_dropout=0.072, dim_head=256)
        if 'model_state' in checkpoint:
            if checkpoint['config']['model'] != 'performance_predictor':
                raise ValueError('The evaluator requires a performance predictor checkpoint')
            params = checkpoint['config']['model_params']
            checkpoint = checkpoint['model_state']
        self.model = ViT(**params).to(device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"[Evaluator] ViT 加载完成: {ckpt_path}")

    @torch.no_grad()
    def predict(self, geometry_norm):
        if isinstance(geometry_norm, np.ndarray):
            geometry_norm = torch.tensor(geometry_norm, dtype=torch.float32)
        geometry_norm = geometry_norm.to(self.device)
        if geometry_norm.dim() == 2:
            x = geometry_norm.unsqueeze(2)
        else:
            x = geometry_norm
        return self.model(x).cpu().numpy()

    @staticmethod
    def diversity_metrics(samples):
        """计算一组生成样本的多样性"""
        param_std = np.std(samples, axis=0)
        if len(samples) >= 2:
            dists = pdist(samples, metric="euclidean")
            avg_dist = float(np.mean(dists))
        else:
            avg_dist = 0.0
        return param_std, avg_dist

    def evaluate_single(self, geom, target_cond, stats, tolerances=None):
        """直接评估全部生成样本（不做 top-k 筛选）"""
        if tolerances is None:
            tolerances = np.array([0.2, 0.01, 0.01])

        pred_norm = self.predict(geom)
        target_batch = np.tile(target_cond, (len(pred_norm), 1))

        lmin, lmax = stats["label_min"], stats["label_max"]
        pred_de = pred_norm * (lmax - lmin) + lmin
        targ_de = target_batch * (lmax - lmin) + lmin

        mse_norm = np.mean((pred_norm - target_batch) ** 2, axis=1)
        mae_de = np.abs(pred_de - targ_de)
        rel_err = mae_de / (np.abs(targ_de) + 1e-12)
        within = np.all(mae_de <= tolerances, axis=1)

        param_std, avg_dist = self.diversity_metrics(geom)

        return {
            "mse_norm": float(np.mean(mse_norm)),
            "mae_de": mae_de.mean(axis=0).tolist(),
            "rel_err": rel_err.mean(axis=0).tolist(),
            "hit_rate": float(np.mean(within)),
            "diversity_std": float(np.mean(param_std)),
            "diversity_dist": avg_dist,
            "pred_de": pred_de,
            "targ_de": targ_de,
        }

    def evaluate_batch(self, sample_fn, conditions, stats,
                       n_samples=10, tolerances=None):
        """对 M 个目标条件批量评估"""
        agg = {"mse": [], "mae": [[], [], []], "rel": [[], [], []],
               "hit": [], "div_std": [], "div_dist": [],
               "pred_de": [], "targ_de": [], "all_geom": []}

        for cond in conditions:
            geom = sample_fn(cond, n_samples)
            m = self.evaluate_single(geom, cond, stats, tolerances)

            agg["mse"].append(m["mse_norm"])
            for d in range(3):
                agg["mae"][d].append(m["mae_de"][d])
                agg["rel"][d].append(m["rel_err"][d])
            agg["hit"].append(m["hit_rate"])
            agg["div_std"].append(m["diversity_std"])
            agg["div_dist"].append(m["diversity_dist"])
            agg["pred_de"].append(m["pred_de"])
            agg["targ_de"].append(m["targ_de"])
            agg["all_geom"].append(geom)

        pred_all = np.vstack(agg["pred_de"])
        targ_all = np.vstack(agg["targ_de"])

        result_dict = {
            "mse_mean": float(np.mean(agg["mse"])),
            "mse_std": float(np.std(agg["mse"])),
            "mae_flow": float(np.mean(agg["mae"][0])),
            "mae_eff": float(np.mean(agg["mae"][1])),
            "mae_pr": float(np.mean(agg["mae"][2])),
            "rel_flow": float(np.mean(agg["rel"][0])),
            "rel_eff": float(np.mean(agg["rel"][1])),
            "rel_pr": float(np.mean(agg["rel"][2])),
            "hit_rate": float(np.mean(agg["hit"])),
            "diversity_std": float(np.mean(agg["div_std"])),
            "diversity_dist": float(np.mean(agg["div_dist"])),
            "pred_denorm": pred_all,
            "target_denorm": targ_all,
            "all_geom": np.vstack(agg["all_geom"]),
        }

        dim_names = ["flow", "eff", "pr"]
        for d, dn in enumerate(dim_names):
            residuals = pred_all[:, d] - targ_all[:, d]
            rmse = float(np.sqrt(np.mean(residuals**2)))
            tgt_range = float(targ_all[:, d].max() - targ_all[:, d].min()) + 1e-12
            nrmse = rmse / tgt_range
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((targ_all[:, d] - targ_all[:, d].mean())**2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            mape = float(np.mean(np.abs(residuals) / (np.abs(targ_all[:, d]) + 1e-12)) * 100)
            result_dict[f"rmse_{dn}"] = rmse
            result_dict[f"nrmse_{dn}"] = float(nrmse)
            result_dict[f"r2_{dn}"] = float(r2)
            result_dict[f"mape_{dn}"] = mape

        return result_dict
