"""Portable predictor HPO using the original Bayesian-search parameter space.

The objective is each trial's best validation MSE. This entrypoint is a
packaging-time port, not a regenerated record of historical experiments.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import optuna
import torch
from models.performance_predictor import train as training
from models.performance_predictor.vit import ViT


def search_space(trial):
    return {
        'dim': trial.suggest_categorical('dim', [128, 256, 512, 1024]),
        'depth': trial.suggest_int('depth', 6, 18),
        'heads': trial.suggest_int('heads', 10, 25),
        'mlp_dim': trial.suggest_categorical('mlp_dim', [128, 256, 512, 1024]),
        'dim_head': trial.suggest_categorical('dim_head', [32, 64, 128, 256, 512]),
        'dropout': trial.suggest_float('dropout', .05, .4),
        'emb_dropout': trial.suggest_float('emb_dropout', .05, .4),
        # Historical spelling retained in the search record.
        'bach_size': trial.suggest_categorical('bach_size', [16, 32, 64]),
    }


def build_model(params):
    architecture = {k: v for k, v in params.items() if k != 'bach_size'}
    return ViT(lenth=1, channels=21, num_classes=3, **architecture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT/'data/parameter_geometry_5200.npy'))
    parser.add_argument('--labels', default=str(ROOT/'data/label_5200.npy'))
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output-dir', default=str(Path(__file__).parent/'results/predictor_hpo'))
    args = parser.parse_args()
    if args.epochs < 1 or args.n_trials < 1:
        parser.error('epochs and n-trials must be positive')
    output = Path(args.output_dir).resolve()
    # A dedicated empty folder prevents replacing a previous optimization.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('output-dir must be empty; use a new directory for a new study')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training.DEVICE = str(device)
    best_overall = float('inf')

    def objective(trial):
        nonlocal best_overall
        params = search_space(trial)
        train_loader, validation_loader = training.load_data(
            args.data, args.labels, batch_size=params['bach_size'], seed=15)
        model = build_model(params).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
        loss_fn = torch.nn.MSELoss()
        best_trial, best_state = float('inf'), None
        for epoch in range(args.epochs):
            training.train_epoch(model, train_loader, loss_fn, optimizer)
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
                val_loss = training.validate(model, validation_loader, loss_fn)[0]
                if val_loss < best_trial:
                    best_trial = val_loss
                    bare = model.module if isinstance(model, torch.nn.DataParallel) else model
                    best_state = {k: v.detach().cpu().clone() for k, v in bare.state_dict().items()}
        if best_trial < best_overall:
            best_overall = best_trial
            torch.save(best_state, output/'best_predictor_state.pth')
            (output/'best_predictor_config.json').write_text(json.dumps(
                {'params':params, 'validation_mse':best_trial, 'trial':trial.number,
                 'split_seed':15, 'epochs':args.epochs}, indent=2), encoding='utf-8')
        del model, optimizer, scheduler, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_trial

    storage = optuna.storages.RDBStorage('sqlite:///' + (output/'study.db').as_posix())
    try:
        study = optuna.create_study(direction='minimize', storage=storage)
        study.optimize(objective, n_trials=args.n_trials)
        print('Best validation MSE:', study.best_value)
        print('Best configuration:', study.best_params)
    finally:
        storage.remove_session()
        storage.engine.dispose()


if __name__ == '__main__':
    main()
