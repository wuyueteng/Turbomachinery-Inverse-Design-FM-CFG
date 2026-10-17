"""Data-free CPU interface checks; random tensors are not research evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.optuna_hpo import create_model
from models.performance_predictor.vit import ViT


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    x, cond = torch.rand(4, 21), torch.rand(4, 3)
    configs = {
        'fm_cfg': dict(hidden_dim=32, n_blocks=1, sampling_steps=2),
        'fm': dict(hidden_dim=32, n_blocks=1, sampling_steps=2),
        'ddpm': dict(hidden_dim=32, n_blocks=1, timesteps=4),
        'ddpm_cfg': dict(hidden_dim=32, n_blocks=1, timesteps=4),
        'ar': dict(dim=32, n_heads=2, n_layers=1),
        'cvae': dict(hidden_dim=32, latent_dim=8, n_layers=1),
        'cinn': dict(hidden_dim=32, n_flows=2),
        'cgan': dict(hidden_dim=32, noise_dim=8, n_g_blocks=1, n_d_blocks=1, n_critic=1),
    }
    for name, params in configs.items():
        model = create_model(name, params, device=torch.device('cpu')).cpu()
        model.train()
        loss = model.training_loss(x, cond)
        assert torch.isfinite(loss).all(), name
        if name != 'cgan':
            loss.backward()
        else:
            assert torch.isfinite(model.training_loss(x, cond)).all()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        model.eval()
        result = model.sample(cond[0], n_samples=3)
        assert result.shape == (3, 21) and np.isfinite(result).all(), name
        print(f'PASS {name}: loss, gradients, sampling {result.shape}')
    model = ViT(lenth=1, channels=21, num_classes=3, dim=32, depth=1,
                heads=2, mlp_dim=32, dim_head=16, dropout=.085, emb_dropout=.072)
    y = model(x.unsqueeze(-1))
    assert y.shape == (4, 3) and torch.isfinite(y).all()
    y.square().mean().backward()
    print('PASS performance_predictor: forward and backward (4, 3)')
    from models.pa_fm_cfg.model import PerformanceAwareFlowMatching
    assert callable(PerformanceAwareFlowMatching)
    print('PASS pa_fm_cfg import; full execution requires the research surrogate checkpoint')


if __name__ == '__main__':
    main()
