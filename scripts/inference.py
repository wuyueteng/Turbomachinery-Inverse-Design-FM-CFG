"""
模型推理脚本 — 加载 HPO 最佳权重, 生成样本并保存结果
=====================================================
与绘图脚本 plot_regression.py 分离, 本脚本只做推理和结果持久化.

Usage:
    python scripts/inference.py                          # 全部模型
    python scripts/inference.py --model ar               # 单模型
    python scripts/inference.py --model ar fm_cfg  # 多个模型
    python scripts/inference.py --n_cond 200 --n_samples 20 --gpu 0"""
import os, sys, json, argparse, csv
import numpy as np
import torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
from shared.data_utils import get_dataloaders, denormalize, GEOM_NAMES, PERF_NAMES
from shared.evaluation import PerformanceEvaluator
from scripts.optuna_hpo import create_model
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ALL_MODELS = ['fm_cfg', 'fm', 'ddpm', 'cinn', 'ar', 'cvae', 'cgan']
DISPLAY = {'fm_cfg': 'cFM-CFG', 'fm': 'FM', 'ddpm': 'DDPM', 'ar': 'cAR', 'cinn': 'cINN', 'cvae': 'cVAE', 'cgan': 'cWGAN-GP'}

def load_best_model(model_name, device):
    """Load an explicitly trained HPO checkpoint; never start training implicitly."""
    hpo_path = os.path.join(MODELS_DIR, model_name, 'results', f'hpo_best_{model_name}.json')
    with open(hpo_path) as f:
        best = json.load(f)
    params = best['best_params']
    model = create_model(model_name, params, device=device).to(device)
    ckpt_path = os.path.join(MODELS_DIR, model_name, 'checkpoints', f'{model_name}_hpo.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f'Missing {ckpt_path}. Train with python scripts/train.py --model {model_name}, then use --run-dir.')
    model.eval()
    if hasattr(model, 'temperature'):
        model.temperature = 0.8
    return (model, params)

def save_results(model_name, result, stats, out_dir):
    """将推理结果保存为 npy / csv / txt"""
    os.makedirs(out_dir, exist_ok=True)
    pred_de = result['pred_denorm']
    targ_de = result['target_denorm']
    all_geom = result['all_geom']
    geom_de = denormalize(all_geom, stats['geom_min'], stats['geom_max'])
    np.save(os.path.join(out_dir, 'geometry_norm.npy'), all_geom)
    np.save(os.path.join(out_dir, 'geometry_denorm.npy'), geom_de)
    np.save(os.path.join(out_dir, 'pred_performance.npy'), pred_de)
    np.save(os.path.join(out_dir, 'target_performance.npy'), targ_de)
    header = GEOM_NAMES + [f'Target_{n}' for n in PERF_NAMES] + [f'Pred_{n}' for n in PERF_NAMES]
    rows = np.hstack([geom_de, targ_de, pred_de])
    csv_path = os.path.join(out_dir, 'inference_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f'{v:.6f}' for v in row])
    dim_names = ['flow', 'eff', 'pr']
    txt_path = os.path.join(out_dir, 'inference_metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {DISPLAY.get(model_name, model_name)}\n')
        f.write(f'Samples: {len(pred_de)}\n')
        f.write('=' * 60 + '\n\n')
        f.write(f"{'Metric':<20s} {'Flow':>12s} {'Efficiency':>12s} {'PR':>12s}\n")
        f.write('-' * 60 + '\n')
        r2 = []
        for d in range(3):
            res = pred_de[:, d] - targ_de[:, d]
            ss_res = np.sum(res ** 2)
            ss_tot = np.sum((targ_de[:, d] - targ_de[:, d].mean()) ** 2) + 1e-12
            r2.append(1 - ss_res / ss_tot)
        f.write(f"{'R2':<20s} {r2[0]:>12.4f} {r2[1]:>12.4f} {r2[2]:>12.4f}\n")
        nrmse = []
        for d in range(3):
            res = pred_de[:, d] - targ_de[:, d]
            rmse = np.sqrt(np.mean(res ** 2))
            tgt_range = targ_de[:, d].max() - targ_de[:, d].min() + 1e-12
            nrmse.append(rmse / tgt_range)
        f.write(f"{'nRMSE':<20s} {nrmse[0]:>12.4f} {nrmse[1]:>12.4f} {nrmse[2]:>12.4f}\n")
        mre = []
        for d in range(3):
            mre.append(np.mean(np.abs(pred_de[:, d] - targ_de[:, d]) / (np.abs(targ_de[:, d]) + 1e-12)) * 100)
        f.write(f"{'MRE (%)':<20s} {mre[0]:>12.2f} {mre[1]:>12.2f} {mre[2]:>12.2f}\n")
        mae = np.mean(np.abs(pred_de - targ_de), axis=0)
        f.write(f"{'MAE':<20s} {mae[0]:>12.4f} {mae[1]:>12.4f} {mae[2]:>12.4f}\n")
        f.write('-' * 60 + '\n')
        f.write(f"{'MSE (norm)':<20s} {result['mse_mean']:>12.6f}\n")
        f.write(f"{'Hit Rate':<20s} {result['hit_rate']:>12.4f}\n")
        f.write(f"{'Diversity (dist)':<20s} {result['diversity_dist']:>12.4f}\n")
        f.write(f"{'Diversity (std)':<20s} {result['diversity_std']:>12.4f}\n")
    metrics = {'model': model_name, 'display_name': DISPLAY.get(model_name, model_name), 'n_samples': int(len(pred_de)), 'mse_norm': result['mse_mean'], 'hit_rate': result['hit_rate'], 'diversity_dist': result['diversity_dist'], 'diversity_std': result['diversity_std']}
    for (d, dn) in enumerate(dim_names):
        metrics[f'r2_{dn}'] = float(r2[d])
        metrics[f'nrmse_{dn}'] = float(nrmse[d])
        metrics[f'mre_{dn}'] = float(mre[d])
        metrics[f'mae_{dn}'] = float(mae[d])
    with open(os.path.join(out_dir, 'inference_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved: {out_dir}/  (npy + csv + txt + json)')

def main():
    parser = argparse.ArgumentParser(description='模型推理: 生成样本并保存结果')
    parser.add_argument('--model', nargs='*', default=None, help='模型名称 (可多个), 不指定则运行全部')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_cond', type=int, default=100, help='测试条件数')
    parser.add_argument('--n_samples', type=int, default=10, help='每个条件的生成样本数')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    models_to_run = args.model if args.model else ALL_MODELS
    print('=' * 60)
    print('  Inference — Generate & Save Results')
    print(f'  Device: {device}')
    print(f'  Models: {models_to_run}')
    print(f'  Conditions: {args.n_cond}, Samples/cond: {args.n_samples}')
    print('=' * 60)
    (_, _, _, info) = get_dataloaders(batch_size=256, use_merged=True)
    stats = info['stats']
    test_label = info['test_label']
    rng = np.random.RandomState(42)
    eval_idx = rng.choice(len(test_label), args.n_cond, replace=False)
    conditions = test_label[eval_idx]
    evaluator = PerformanceEvaluator(str(device))
    for (i, m) in enumerate(models_to_run):
        hpo_path = os.path.join(MODELS_DIR, m, 'results', f'hpo_best_{m}.json')
        if not os.path.exists(hpo_path):
            print(f'\n  [{m}] No HPO result found, skip')
            continue
        print(f'\n[{i + 1}/{len(models_to_run)}] {DISPLAY.get(m, m)} ...')
        (model, params) = load_best_model(m, device)

        def sample_fn(c, n, _m=model, _name=m):
            if _name in ['ar', 'cinn']:
                return _m.sample(c, n_samples=n, temperature=0.8)
            return _m.sample(c, n_samples=n)
        result = evaluator.evaluate_batch(sample_fn, conditions, stats, n_samples=args.n_samples)
        out_dir = os.path.join(MODELS_DIR, m, 'results', 'inference')
        save_results(m, result, stats, out_dir)
        print(f"  MSE={result['mse_mean']:.6f}  HR={result['hit_rate']:.4f}  R2=[{result['r2_flow']:.4f},{result['r2_eff']:.4f},{result['r2_pr']:.4f}]")
        del model
        torch.cuda.empty_cache()
    print('\nDone!')
if __name__ == '__main__':
    if not any(a == '--model' or a.startswith('--model=') for a in sys.argv):
        from scripts.run_inference import main as run_main
        run_main()
    else:
        main()
