"""
Conditional Flow Matching (CFM)
================================
基于 Lipman et al. (2023) 的条件流匹配方法。
核心思想：学习从噪声到数据的速度场 v_θ(x_t, t, c)，
通过 ODE 积分进行采样，无需噪声调度。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLMResBlock(nn.Module):
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return x + self.net(h)


class VelocityNet(nn.Module):
    """预测速度场 v(x_t, t, c)"""
    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 time_dim=256, n_blocks=8, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Linear(geom_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim, time_dim, dropout) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, geom_dim),
        )

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        tc = t_emb + c_emb
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, tc)
        return self.output_proj(h)


class ConditionalFlowMatching(nn.Module):
    """条件流匹配模型"""

    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 n_blocks=8, sigma_min=1e-4, p_uncond=0.15,
                 guidance_scale=2.0, sampling_steps=100, dropout=0.1):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.sampling_steps = sampling_steps

        self.net = VelocityNet(
            geom_dim=geom_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
        )

    def training_loss(self, x_1, cond):
        B = x_1.shape[0]
        device = x_1.device
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        target = x_1 - x_0
        mask = torch.rand(B, device=device) < self.p_uncond
        cond_masked = cond.clone()
        cond_masked[mask] = 0.0
        v_pred = self.net(x_t, t, cond_masked)
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, cond, n_samples=128, guidance_scale=None, steps=None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        if steps is None:
            steps = self.sampling_steps
        device = next(self.parameters()).device
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)
        null_cond = torch.zeros_like(cond_batch)
        dt = 1.0 / steps
        x = torch.randn(n_samples, self.geom_dim, device=device)
        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=device)
            v_cond = self.net(x, t, cond_batch)
            v_uncond = self.net(x, t, null_cond)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + v * dt
        return x.clamp(0, 1).cpu().numpy()

    @torch.no_grad()
    def sample_unconditional(self, n_samples=256):
        device = next(self.parameters()).device
        null_cond = torch.zeros(n_samples, self.cond_dim, device=device)
        dt = 1.0 / self.sampling_steps
        x = torch.randn(n_samples, self.geom_dim, device=device)
        for i in range(self.sampling_steps):
            t = torch.full((n_samples,), i * dt, device=device)
            v = self.net(x, t, null_cond)
            x = x + v * dt
        return x.clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = ConditionalFlowMatching()
    x = torch.randn(4, 21).clamp(0, 1)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
