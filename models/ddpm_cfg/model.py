"""
DDPM-CFG — 基于 DDPM 最佳配置 + Classifier-Free Guidance
=========================================================
与 DDPM (`models/ddpm/`) 共享相同的 NoisePredictor 架构, 区别:
  - 训练: 条件 dropout (p_uncond), 随机置零条件向量
  - 采样: 双次前向 + CFG 引导: eps = eps_uncond + w*(eps_cond - eps_uncond)

用于消融实验: 验证 CFG 对 DDPM 条件生成质量的贡献.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 复用 DDPM 的 NoisePredictor 和 schedule (相同架构)
from models.ddpm.model import NoisePredictor, cosine_beta_schedule


class DDPM_CFG(nn.Module):
    """DDPM + Classifier-Free Guidance (基于 DDPM 最佳架构)"""

    def __init__(self, geom_dim=21, cond_dim=3, timesteps=450,
                 hidden_dim=1024, n_blocks=7, dropout=0.037,
                 p_uncond=0.15, guidance_scale=2.0):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale

        self.net = NoisePredictor(
            geom_dim=geom_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
        )

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * x_0 + sqrt_omac * noise, noise

    def training_loss(self, x_0, cond):
        """训练损失: 带条件 dropout 的噪声预测"""
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)
        # CFG: 随机置零条件
        mask = torch.rand(B, device=x_0.device) < self.p_uncond
        cond_masked = cond.clone()
        cond_masked[mask] = 0.0
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.net(x_t, t, cond_masked)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond, n_samples=128, guidance_scale=None):
        """采样: 双前向 CFG 引导"""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        device = self.betas.device

        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)
        null_cond = torch.zeros_like(cond_batch)

        x = torch.randn(n_samples, self.geom_dim, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            eps_cond = self.net(x, t, cond_batch)
            eps_uncond = self.net(x, t, null_cond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            beta = self.betas[i]

            mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * eps)
            if i > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = DDPM_CFG()
    x = torch.randn(4, 21).clamp(0, 1)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
