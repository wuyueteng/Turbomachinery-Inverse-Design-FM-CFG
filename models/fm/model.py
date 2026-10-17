"""
Conditional Flow Matching (FM) — 无 CFG 消融版本
=================================================
与 FM-CFG 共享相同的 VelocityNet 架构, 区别:
  - 训练: 不做条件 dropout (p_uncond=0), 始终使用真实条件
  - 采样: 直接使用条件速度场, 无 guidance scale 放大

用于消融实验: 验证 Classifier-Free Guidance 对条件生成质量的贡献.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 复用 FM-CFG 的 VelocityNet (相同架构)
from models.fm_cfg.model import VelocityNet


class ConditionalFM(nn.Module):
    """条件流匹配模型 (无 CFG)"""

    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 n_blocks=8, sigma_min=1e-4, sampling_steps=100, dropout=0.1):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.sigma_min = sigma_min
        self.sampling_steps = sampling_steps

        self.net = VelocityNet(
            geom_dim=geom_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
        )

    def training_loss(self, x_1, cond):
        """标准条件流匹配损失, 无条件 dropout"""
        B = x_1.shape[0]
        device = x_1.device
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        target = x_1 - x_0
        # 无 condition dropout: 始终传入真实条件
        v_pred = self.net(x_t, t, cond)
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, cond, n_samples=128, steps=None):
        if steps is None:
            steps = self.sampling_steps
        device = next(self.parameters()).device

        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)

        dt = 1.0 / steps
        x = torch.randn(n_samples, self.geom_dim, device=device)

        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=device)
            # 无 CFG: 直接使用条件速度场, 单次前向传播
            v = self.net(x, t, cond_batch)
            x = x + v * dt

        return x.clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = ConditionalFM()
    x = torch.randn(4, 21).clamp(0, 1)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
