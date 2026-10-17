"""
Optuna 超参数寻优 — 7 种生成模型 (公平对比版 v2)
==================================================
嵌套两阶段法:
  内层: 用各模型自身损失训练到收敛 (val_loss 早停, cGAN 不早停)
  外层: 训练完成后, 在验证集上用下游 MSE (ViT 预测 vs 目标) 作为 Optuna 目标

公平性保证:
  - 统一 HPO 目标: downstream MSE (所有模型相同的评估标准)
  - 统一优化器: Adam + CosineAnnealingLR (cGAN 内部自管理 opt_g + opt_d)
  - 统一计算预算: max_epochs, batch_size
  - 早停: 非GAN模型 val_loss patience; cGAN 不早停
  - 统一剪枝: MedianPruner, 定期下游 MSE 报告
  - 统一搜索: TPESampler(seed=42), 相同 n_trials

Usage:
    python scripts/optuna_hpo.py                      # 全部 6 个模型, 30 trials
    python scripts/optuna_hpo.py --model fm_cfg        # 单模型
    python scripts/optuna_hpo.py --model cgan          # 单模型
    python scripts/optuna_hpo.py --n_trials 50         # 自定义试验数
    python scripts/optuna_hpo.py --retrain             # HPO 后用最佳超参重训并保存权重"""
import os, sys, json, copy, argparse, gc
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
from shared.data_utils import get_dataloaders, load_norm_stats
from shared.evaluation import PerformanceEvaluator
from models.fm_cfg import ConditionalFlowMatching
from models.cvae import ConditionalVAE
from models.cgan import ConditionalGAN
from models.ar import ConditionalAutoregressive
from models.cinn import ConditionalGlow
from models.ddpm.model import ConditionalDDPM
from models.fm.model import ConditionalFM

def _get_device(gpu_id=None):
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = _get_device()
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ALL_MODELS = ['ddpm', 'fm_cfg', 'cvae', 'cgan', 'ar', 'cinn']
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 50
DEFAULT_BATCH = 256
MIN_EPOCHS = 100
GRAD_CLIP = 1.0
SCHEDULER_ETA_MIN = 1e-06
EVAL_N_CONDITIONS = 50
EVAL_N_SAMPLES = 5
EVAL_INTERVAL = 15

def _result_dir(model_name):
    d = os.path.join(MODELS_DIR, model_name, 'results')
    os.makedirs(d, exist_ok=True)
    return d

def _ckpt_dir(model_name):
    d = os.path.join(MODELS_DIR, model_name, 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d

def define_search_space(trial, model_name):
    if model_name == 'ddpm':
        return {'timesteps': trial.suggest_int('timesteps', 100, 500, step=50), 'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]), 'n_blocks': trial.suggest_int('n_blocks', 4, 12), 'dropout': trial.suggest_float('dropout', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    elif model_name == 'fm_cfg':
        return {'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]), 'n_blocks': trial.suggest_int('n_blocks', 4, 12), 'p_uncond': trial.suggest_float('p_uncond', 0.05, 0.3), 'guidance_scale': trial.suggest_float('guidance_scale', 0.5, 5.0), 'sampling_steps': trial.suggest_int('sampling_steps', 50, 200, step=25), 'dropout': trial.suggest_float('dropout', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    elif model_name == 'fm':
        return {'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]), 'n_blocks': trial.suggest_int('n_blocks', 4, 12), 'sampling_steps': trial.suggest_int('sampling_steps', 50, 200, step=25), 'dropout': trial.suggest_float('dropout', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    elif model_name == 'cvae':
        return {'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]), 'latent_dim': trial.suggest_int('latent_dim', 8, 64, step=8), 'n_layers': trial.suggest_int('n_layers', 2, 8), 'beta': trial.suggest_float('beta', 0.01, 2.0), 'dropout': trial.suggest_float('dropout', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    elif model_name == 'cgan':
        return {'noise_dim': trial.suggest_int('noise_dim', 32, 128, step=16), 'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]), 'n_g_blocks': trial.suggest_int('n_g_blocks', 3, 8), 'n_d_blocks': trial.suggest_int('n_d_blocks', 2, 6), 'n_critic': trial.suggest_int('n_critic', 3, 10), 'lambda_gp': trial.suggest_float('lambda_gp', 1.0, 50.0), 'lr_g': trial.suggest_float('lr_g', 1e-05, 0.001, log=True), 'lr_d': trial.suggest_float('lr_d', 1e-05, 0.001, log=True)}
    elif model_name == 'ar':
        dim = trial.suggest_categorical('dim', [128, 256, 512])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        while dim % n_heads != 0:
            n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        return {'dim': dim, 'n_heads': n_heads, 'n_layers': trial.suggest_int('n_layers', 3, 10), 'dropout': trial.suggest_float('dropout', 0.0, 0.3), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    elif model_name == 'cinn':
        return {'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]), 'n_flows': trial.suggest_int('n_flows', 4, 16), 'lr': trial.suggest_float('lr', 1e-05, 0.001, log=True)}
    raise ValueError(f'Unknown model: {model_name}')

def create_model(model_name, params, device=None):
    """创建模型, 过滤掉纯训练参数"""
    exclude = {'lr'}
    if model_name != 'cgan':
        exclude.update({'lr_g', 'lr_d'})
    if model_name not in ('ar', 'cinn'):
        exclude.add('temperature')
    p = {k: v for (k, v) in params.items() if k not in exclude}
    factories = {'ddpm': lambda : ConditionalDDPM(geom_dim=21, cond_dim=3, **p), 'fm_cfg': lambda : ConditionalFlowMatching(geom_dim=21, cond_dim=3, **p), 'fm': lambda : ConditionalFM(geom_dim=21, cond_dim=3, **p), 'cvae': lambda : ConditionalVAE(geom_dim=21, cond_dim=3, **p), 'cgan': lambda : ConditionalGAN(geom_dim=21, cond_dim=3, **p), 'ar': lambda : ConditionalAutoregressive(geom_dim=21, cond_dim=3, **p), 'cinn': lambda : ConditionalGlow(geom_dim=21, cond_dim=3, **p)}
    return factories[model_name]()

def train_one_epoch(model, loader, optimizer, model_name, epoch=0):
    """训练一轮
    - cGAN: training_loss 内部自管理 opt_g/opt_d, 外部 optimizer 不使用
    - 其他: 标准 loss.backward() + optimizer.step()
    """
    model.train()
    (total_loss, n) = (0, 0)
    is_gan = model_name == 'cgan'
    for batch in loader:
        geom = batch['geometry'].to(DEVICE)
        label = batch['label'].to(DEVICE)
        loss = model.training_loss(geom, label)
        if torch.isnan(loss):
            return float('nan')
        if not is_gan:
            optimizer.zero_grad()
            loss.backward()
            trainable = model.parameters()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
        total_loss += loss.item() * geom.shape[0]
        n += geom.shape[0]
    return total_loss / n

@torch.no_grad()
def validate(model, loader, model_name, epoch=0):
    """验证损失
    - cGAN: G loss on val set
    - 其他: training_loss
    """
    model.eval()
    (total_loss, n) = (0, 0)
    is_gan = model_name == 'cgan'
    for batch in loader:
        geom = batch['geometry'].to(DEVICE)
        label = batch['label'].to(DEVICE)
        if is_gan:
            z = torch.randn(geom.shape[0], model.noise_dim, device=DEVICE)
            fake = model.generator(z, label)
            g_loss = -model.discriminator(fake, label).mean()
            total_loss += g_loss.item() * geom.shape[0]
        else:
            loss = model.training_loss(geom, label)
            total_loss += loss.item() * geom.shape[0]
        n += geom.shape[0]
    return total_loss / n

@torch.no_grad()
def downstream_mse(model, evaluator, conditions, stats):
    """在验证集条件上采样, 用 ViT 预测性能, 计算与目标的归一化 MSE"""
    model.eval()
    mse_list = []
    for cond in conditions:
        geom = model.sample(cond, n_samples=EVAL_N_SAMPLES)
        result = evaluator.evaluate_single(geom, cond, stats)
        mse_list.append(result['mse_norm'])
    return float(np.mean(mse_list))

def objective(trial, model_name, train_loader, val_loader, evaluator, eval_conditions, stats, epochs, patience):
    """
    嵌套两阶段目标函数:
      内层: 用各模型自身损失训练, val_loss 控制早停 (cGAN 不早停)
      外层: 训练完成后 + 定期, 用下游 MSE 作为 Optuna 目标值

    所有模型共用同一个函数, 共用同一个评估标准 (downstream MSE)
    """
    params = define_search_space(trial, model_name)
    try:
        model = create_model(model_name, params).to(DEVICE)
    except Exception as e:
        print(f'    [!] {model_name} creation failed: {e}')
        raise optuna.TrialPruned()
    n_params = sum((p.numel() for p in model.parameters()))
    trial.set_user_attr('n_params', n_params)
    is_gan = model_name == 'cgan'
    if is_gan:
        optimizer = None
        scheduler = None
    else:
        lr = params.get('lr', 0.0001)
        optimizer = Adam(model.parameters(), lr=lr)
        pa_epochs = epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=pa_epochs, eta_min=SCHEDULER_ETA_MIN)
    max_ep = epochs
    pa_patience = patience
    pa_min = MIN_EPOCHS
    best_val = float('inf')
    best_state = None
    no_improve = 0
    best_mse = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'downstream_mse': []}
    try:
        for epoch in range(1, max_ep + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, model_name, epoch=epoch)
            if np.isnan(train_loss):
                raise optuna.TrialPruned()
            if scheduler is not None:
                scheduler.step()
            val_loss = validate(model, val_loader, model_name, epoch=epoch)
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if epoch % EVAL_INTERVAL == 0:
                interim_mse = downstream_mse(model, evaluator, eval_conditions, stats)
                history['downstream_mse'].append({'epoch': epoch, 'mse': interim_mse})
                if interim_mse < best_mse:
                    best_mse = interim_mse
                trial.report(interim_mse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if not is_gan and no_improve >= pa_patience and (epoch >= pa_min):
                break
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
        raise
    if best_state is not None:
        model.load_state_dict(best_state)
    final_mse = downstream_mse(model, evaluator, eval_conditions, stats)
    history['downstream_mse'].append({'epoch': epoch, 'mse': final_mse, 'final': True})
    del model
    if optimizer is not None:
        del optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    trial.set_user_attr('history', history)
    trial.set_user_attr('stopped_epoch', len(history['train_loss']))
    trial.set_user_attr('final_mse', final_mse)
    return final_mse

def run_hpo_for_model(model_name, train_loader, val_loader, evaluator, eval_conditions, stats, n_trials=40, epochs=300, patience=50):
    db_path = os.path.join(_result_dir(model_name), f'optuna_hpo_{model_name}.db')
    storage = f'sqlite:///{db_path}'
    study = optuna.create_study(study_name=f'hpo_{model_name}', direction='minimize', sampler=TPESampler(seed=42), storage=storage, load_if_exists=True)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining == 0:
        print(f'  [{model_name}] done {completed} trials, skip')
    else:
        print(f'  [{model_name}] {completed} done, {remaining} remaining...')
        study.optimize(lambda trial: objective(trial, model_name, train_loader, val_loader, evaluator, eval_conditions, stats, epochs, patience), n_trials=remaining, show_progress_bar=True)
    if study.best_trial is not None:
        best_trial = study.best_trial
        res_dir = _result_dir(model_name)
        result = {'model': model_name, 'best_params': study.best_params, 'best_downstream_mse': float(study.best_value), 'best_n_params': best_trial.user_attrs.get('n_params', 'N/A'), 'stopped_epoch': best_trial.user_attrs.get('stopped_epoch', 'N/A'), 'n_trials': len(study.trials), 'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]), 'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]), 'hpo_config': {'strategy': 'TPE (seed=42)', 'objective': 'downstream_mse (ViT pred vs target on val conditions)', 'eval_n_conditions': EVAL_N_CONDITIONS, 'eval_n_samples': EVAL_N_SAMPLES, 'optimizer': 'Adam' if model_name != 'cgan' else 'Adam (internal opt_g + opt_d)', 'scheduler': 'CosineAnnealingLR' if model_name != 'cgan' else 'None (internal)', 'early_stopping': f'patience={patience}, min_epoch={MIN_EPOCHS}' if model_name != 'cgan' else 'disabled (adversarial)', 'max_epochs': epochs, 'patience': patience if model_name != 'cgan' else 'N/A', 'min_epochs': MIN_EPOCHS, 'batch_size': DEFAULT_BATCH, 'grad_clip': GRAD_CLIP}}
        with open(os.path.join(res_dir, f'hpo_best_{model_name}.json'), 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        trial_history = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                trial_history.append({'number': t.number, 'value': float(t.value), 'params': t.params, 'n_params': t.user_attrs.get('n_params'), 'stopped_epoch': t.user_attrs.get('stopped_epoch'), 'final_mse': t.user_attrs.get('final_mse'), 'history': t.user_attrs.get('history')})
        with open(os.path.join(res_dir, f'hpo_history_{model_name}.json'), 'w') as f:
            json.dump(trial_history, f, indent=2, ensure_ascii=False)
        summary_lines = [f'Model: {model_name}', f'Best downstream MSE: {study.best_value:.6f}', f'Best params: {study.best_params}', f'Best trial: #{best_trial.number}', f"Stopped epoch: {best_trial.user_attrs.get('stopped_epoch', 'N/A')}", f"N params: {best_trial.user_attrs.get('n_params', 'N/A'):,}", f"Completed/Pruned: {result['n_completed']}/{result['n_pruned']}", '', 'All trials (sorted by MSE):', f"{'Trial':>6s}  {'MSE':>12s}  {'Epoch':>6s}  {'Params':>12s}", '-' * 42]
        sorted_trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
        for t in sorted_trials:
            n_p = t.user_attrs.get('n_params', 0)
            ep = t.user_attrs.get('stopped_epoch', '?')
            summary_lines.append(f'  #{t.number:<4d}  {t.value:>12.6f}  {str(ep):>6s}  {n_p:>12,}')
        with open(os.path.join(res_dir, f'hpo_summary_{model_name}.txt'), 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f'\n  [{model_name}] Best downstream MSE: {study.best_value:.6f}')
        print(f'  [{model_name}] Best params: {study.best_params}')
        print(f"  [{model_name}] Stopped @ epoch: {result['stopped_epoch']}")
        print(f"  [{model_name}] Completed/Pruned: {result['n_completed']}/{result['n_pruned']}")
    return study

def retrain_best(model_name, train_loader, val_loader, epochs=300, patience=30):
    hpo_path = os.path.join(_result_dir(model_name), f'hpo_best_{model_name}.json')
    if not os.path.exists(hpo_path):
        print(f'  [{model_name}] 无 HPO 结果, 跳过')
        return
    with open(hpo_path) as f:
        best = json.load(f)
    params = best['best_params']
    model = create_model(model_name, params).to(DEVICE)
    is_gan = model_name == 'cgan'
    if is_gan:
        (optimizer, scheduler) = (None, None)
    else:
        lr = params.get('lr', 0.0001)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=SCHEDULER_ETA_MIN)
    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, model_name)
        val_loss = validate(model, val_loader, model_name)
        if scheduler is not None:
            scheduler.step()
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 50 == 0:
            print(f'    epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}')
        if not is_gan and no_improve >= patience and (epoch >= MIN_EPOCHS):
            print(f'    早停 @ epoch {epoch}')
            break
    ckpt_path = os.path.join(_ckpt_dir(model_name), f'{model_name}_hpo.pth')
    torch.save(best_state, ckpt_path)
    print(f'  [{model_name}] 权重: {ckpt_path}')
    hist_path = os.path.join(_result_dir(model_name), f'retrain_history_{model_name}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'  [{model_name}] 训练历史: {hist_path}')
    del model
    if optimizer is not None:
        del optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Optuna HPO (公平对比)')
    parser.add_argument('--predictor-run', help='Run directory from scripts/train.py --model performance_predictor')
    parser.add_argument('--model', type=str, default='all', choices=['all'] + ALL_MODELS)
    parser.add_argument('--n_trials', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--retrain', action='store_true', help='HPO 完成后用最佳超参重训并保存权重')
    parser.add_argument('--gpu', type=int, default=None, help='指定 GPU 编号 (0, 1, ...)')
    args = parser.parse_args()
    global DEVICE
    DEVICE = _get_device(args.gpu)
    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]
    print('=' * 65)
    print('  Optuna HPO v2 — Downstream MSE Objective')
    print('=' * 65)
    print(f"  Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'  Device:       {DEVICE}')
    print(f'  Models:       {models_to_run}')
    print(f'  Sampler:      TPE (seed=42)')
    print(f'  Pruner:       None (evaluate only after training)')
    print(f'  Trials:       {args.n_trials} per model')
    print(f'  Max epochs:   {args.epochs}')
    print(f'  Early stop:   val_loss patience={args.patience}, min={MIN_EPOCHS} (cGAN disabled)')
    print(f'  HPO target:   downstream MSE ({EVAL_N_CONDITIONS} conds x {EVAL_N_SAMPLES} samples)')
    print(f'  Eval interval: every {EVAL_INTERVAL} epochs')
    print(f'  Optimizer:    Adam + CosineAnnealingLR (cGAN internal)')
    print(f'  Batch size:   {DEFAULT_BATCH}')
    print('=' * 65)
    print('\n[1/4] Loading data...')
    (train_loader, val_loader, test_loader, info) = get_dataloaders(batch_size=DEFAULT_BATCH)
    stats = info['stats']
    rng = np.random.RandomState(42)
    n_eval = min(EVAL_N_CONDITIONS, info['n_val'])
    eval_idx = rng.choice(info['n_val'], n_eval, replace=False)
    eval_conditions = info['val_label'][eval_idx]
    print(f"  Eval conditions: {n_eval} x {EVAL_N_SAMPLES} samples (from val set {info['n_val']})")
    print('[2/4] Loading ViT evaluator...')
    evaluator = PerformanceEvaluator(str(DEVICE), checkpoint_path=os.path.join(args.predictor_run, 'best.pt') if args.predictor_run else None)
    print(f'\n[3/4] HPO ({len(models_to_run)} models)...')
    studies = {}
    for model_name in models_to_run:
        print(f"\n{'=' * 60}")
        print(f'  HPO: {model_name}')
        print(f"{'=' * 60}")
        studies[model_name] = run_hpo_for_model(model_name, train_loader, val_loader, evaluator, eval_conditions, stats, n_trials=args.n_trials, epochs=args.epochs, patience=args.patience)
    print(f'\n[4/4] Summary')
    print('=' * 85)
    print(f"{'Model':<16s} {'Best MSE':>14s} {'Completed':>10s} {'Pruned':>8s} {'Epoch':>6s} {'Params':>10s}")
    print('-' * 85)
    for model_name in models_to_run:
        path = os.path.join(_result_dir(model_name), f'hpo_best_{model_name}.json')
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            n_p = r.get('best_n_params', 'N/A')
            n_p_str = f'{n_p:,}' if isinstance(n_p, int) else str(n_p)
            ep = r.get('stopped_epoch', 'N/A')
            best_mse = r.get('best_downstream_mse', r.get('best_val_loss', 0))
            print(f"{model_name:<16s} {best_mse:>14.6f} {r['n_completed']:>10d} {r['n_pruned']:>8d} {str(ep):>6s} {n_p_str:>10s}")
    print('=' * 85)
    if args.retrain:
        print('\n[Retrain] Retrain with best params...')
        for model_name in models_to_run:
            print(f'\n--- {model_name} ---')
            retrain_best(model_name, train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print(f'\nResults: models/{{model}}/results/hpo_best_*.json')
    print(f'History: models/{{model}}/results/hpo_history_*.json')
    print('Done!')
if __name__ == '__main__':
    main()
