"""
用 HPO 最佳超参在测试集上评估 7 个模型 + 绘制散点图
=====================================================
Usage:
    python scripts/eval_hpo_best.py --gpu 0"""
import os, sys, json, argparse
import numpy as np
import torch
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.data_utils import get_dataloaders
from shared.evaluation import PerformanceEvaluator
from scripts.optuna_hpo import create_model
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
FIG_DIR = os.path.join(ROOT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'mathtext.fontset': 'stix'})
ALL_MODELS = ['fm_cfg', 'fm', 'ddpm', 'cinn', 'ar', 'cvae', 'cgan']
DISPLAY = {'fm_cfg': 'cFM-CFG', 'fm': 'FM', 'ddpm': 'DDPM', 'ar': 'cAR', 'cinn': 'cINN', 'cvae': 'cVAE', 'cgan': 'cWGAN-GP'}
COLORS = {'fm_cfg': '#9B59B6', 'fm': '#E67E22', 'ddpm': '#2C3E50', 'ar': '#3498DB', 'cinn': '#7F8C8D', 'cvae': '#1ABC9C', 'cgan': '#F39C12'}
PERF_NAMES = {0: 'Mass flow rate [kg/s]', 1: 'Isentropic efficiency [-]', 2: 'Total pressure ratio [-]'}
PERF_ORDER = [0, 2, 1]
LABEL_RANGES = {0: 3.7483, 1: 0.071, 2: 0.3261}

def load_best_model(model_name, device):
    hpo_path = os.path.join(MODELS_DIR, model_name, 'results', f'hpo_best_{model_name}.json')
    with open(hpo_path) as f:
        best = json.load(f)
    params = best['best_params']
    model = create_model(model_name, params, device=device).to(device)
    ckpt_path = os.path.join(MODELS_DIR, model_name, 'checkpoints', f'{model_name}_hpo.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f'Missing {ckpt_path}. Train explicitly using scripts/train.py or HPO --retrain.')
    model.eval()
    if hasattr(model, 'temperature'):
        model.temperature = 0.8
    return (model, params)

def retrain_with_best_params(model_name, model, device):
    """快速重训并保存权重"""
    import copy
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    hpo_path = os.path.join(MODELS_DIR, model_name, 'results', f'hpo_best_{model_name}.json')
    with open(hpo_path) as f:
        best = json.load(f)
    params = best['best_params']
    is_gan = model_name == 'cgan'
    (train_loader, val_loader, _, info) = get_dataloaders(batch_size=256, use_merged=True)
    lr = params.get('lr', 0.0001)
    (epochs, patience, min_epochs) = (300, 50, 100)
    if is_gan:
        (optimizer, scheduler) = (None, None)
    else:
        trainable = model.parameters()
        optimizer = Adam(trainable, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-06)
    (best_val, best_state, no_improve) = (float('inf'), None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            (g, l) = (batch['geometry'].to(device), batch['label'].to(device))
            loss = model.training_loss(g, l)
            if not is_gan:
                optimizer.zero_grad()
                loss.backward()
                tp = model.parameters()
                nn.utils.clip_grad_norm_(tp, 1.0)
                optimizer.step()
        if scheduler:
            scheduler.step()
        model.eval()
        (vl, n) = (0.0, 0)
        with torch.no_grad():
            for batch in val_loader:
                (g, l) = (batch['geometry'].to(device), batch['label'].to(device))
                if is_gan:
                    z = torch.randn(g.shape[0], model.noise_dim, device=device)
                    fake = model.generator(z, l)
                    vl += (-model.discriminator(fake, l).mean()).item() * g.shape[0]
                else:
                    vl += model.training_loss(g, l).item() * g.shape[0]
                n += g.shape[0]
        vl /= n
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 50 == 0:
            print(f'    epoch {epoch}  val={vl:.6f}')
        if not is_gan and no_improve >= patience and (epoch >= min_epochs):
            print(f'    early stop @ {epoch}')
            break
    ckpt_path = os.path.join(MODELS_DIR, model_name, 'checkpoints', f'{model_name}_hpo.pth')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    print(f'    saved: {ckpt_path}')

def plot_scatter(results, fig_dir):
    models = [m for m in ALL_MODELS if m in results]
    n = len(models)
    (fig, axes) = plt.subplots(n, 3, figsize=(16, 3.2 * n))
    for (i, m) in enumerate(models):
        r = results[m]
        (pred, targ) = (r['pred_denorm'], r['target_denorm'])
        for (j, d) in enumerate(PERF_ORDER):
            ax = axes[i, j]
            ax.scatter(targ[:, d], pred[:, d], alpha=0.35, s=10, color=COLORS[m], edgecolors='none', zorder=2)
            lo = min(targ[:, d].min(), pred[:, d].min())
            hi = max(targ[:, d].max(), pred[:, d].max())
            mg = (hi - lo) * 0.05
            ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], 'k--', lw=0.8, alpha=0.5, zorder=1)
            ax.set_xlim(lo - mg, hi + mg)
            ax.set_ylim(lo - mg, hi + mg)
            ax.set_aspect('equal')
            res = pred[:, d] - targ[:, d]
            ss_res = np.sum(res ** 2)
            ss_tot = np.sum((targ[:, d] - targ[:, d].mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            nrmse = np.sqrt(np.mean(res ** 2)) / LABEL_RANGES[d]
            mre = np.mean(np.abs(res) / (np.abs(targ[:, d]) + 1e-12)) * 100
            ax.text(0.05, 0.95, f'$R^2$={r2:.4f}\nnRMSE={nrmse:.4f}\nMRE={mre:.2f}%', transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='gray', lw=0.5))
            if i == 0:
                ax.set_title(PERF_NAMES[d], fontsize=15)
            if j == 0:
                ax.set_ylabel(f'{DISPLAY[m]}\nPredicted', fontsize=13)
            if i == n - 1:
                ax.set_xlabel('Target', fontsize=13)
            ax.tick_params(direction='in', length=3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'scatter_7models_hpo.png'), dpi=600)
    plt.close(fig)
    print('  Saved scatter_7models_hpo.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_cond', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('  Evaluate HPO Best Models on Test Set')
    print(f'  Device: {device}')
    print('=' * 60)
    (_, _, _, info) = get_dataloaders(batch_size=256, use_merged=True)
    stats = info['stats']
    test_label = info['test_label']
    rng = np.random.RandomState(42)
    eval_idx = rng.choice(len(test_label), args.n_cond, replace=False)
    conditions = test_label[eval_idx]
    evaluator = PerformanceEvaluator(str(device))
    results = {}
    metrics = {}
    for (i, m) in enumerate(ALL_MODELS):
        hpo_path = os.path.join(MODELS_DIR, m, 'results', f'hpo_best_{m}.json')
        if not os.path.exists(hpo_path):
            print(f'  [{m}] No HPO result, skip')
            continue
        print(f'\n[{i + 1}/{len(ALL_MODELS)}] {DISPLAY[m]}...')
        (model, params) = load_best_model(m, device)

        def sample_fn(c, n, _m=model, _name=m):
            if _name in ['ar', 'cinn']:
                return _m.sample(c, n_samples=n, temperature=0.8)
            return _m.sample(c, n_samples=n)
        r = evaluator.evaluate_batch(sample_fn, conditions, stats, n_samples=args.n_samples)
        results[m] = r
        mre = (r['rel_flow'] + r['rel_eff'] + r['rel_pr']) / 3 * 100
        nrmse = np.mean([r['nrmse_flow'], r['nrmse_eff'], r['nrmse_pr']])
        r2_avg = np.mean([r['r2_flow'], r['r2_eff'], r['r2_pr']])
        metrics[m] = {'mse': r['mse_mean'], 'nrmse': nrmse, 'mre': mre, 'r2_avg': r2_avg, 'r2_flow': r['r2_flow'], 'r2_eff': r['r2_eff'], 'r2_pr': r['r2_pr'], 'hit_rate': r['hit_rate'], 'diversity': r['diversity_dist'], 'params': params}
        print(f"  MSE={r['mse_mean']:.6f}  HR={r['hit_rate']:.4f}  R2=[{r['r2_flow']:.4f},{r['r2_eff']:.4f},{r['r2_pr']:.4f}]  MRE={mre:.2f}%")
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(FIG_DIR, 'eval_hpo_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('\n' + '=' * 120)
    print(f"{'Model':<12s} {'MSE':>8s} {'nRMSE':>8s} {'MRE%':>8s} {'R2_flow':>8s} {'R2_eff':>8s} {'R2_pr':>8s} {'HitRate':>8s}")
    print('-' * 120)
    for m in ALL_MODELS:
        if m not in metrics:
            continue
        mt = metrics[m]
        print(f"{DISPLAY[m]:<12s} {mt['mse']:>8.5f} {mt['nrmse']:>8.4f} {mt['mre']:>7.2f}% {mt['r2_flow']:>8.4f} {mt['r2_eff']:>8.4f} {mt['r2_pr']:>8.4f} {mt['hit_rate']:>8.4f}")
    print('=' * 120)
    plot_scatter(results, FIG_DIR)
    print('\nDone!')
if __name__ == '__main__':
    main()
