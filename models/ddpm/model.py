"""
Conditional DDPM (without Classifier-Free Guidance)
=====================================================
Standard conditional denoising diffusion probabilistic model.
Differences from DDPM-CFG:
  - Training: condition is always provided (no random masking, p_uncond=0)
  - Sampling: single forward pass per step (no CFG mixing, no guidance_scale)
  - Same architecture: SinusoidalTimeEmbedding + FiLM ResBlocks + cosine schedule

This serves as an ablation baseline to isolate the effect of CFG.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""
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
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class FiLMResBlock(nn.Module):
    """Feature-wise Linear Modulation residual block."""
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


class NoisePredictor(nn.Module):
    """Residual MLP noise predictor with FiLM conditioning."""
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


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class ConditionalDDPM(nn.Module):
    """Conditional DDPM without Classifier-Free Guidance."""

    def __init__(self, geom_dim=21, cond_dim=3, timesteps=500,
                 hidden_dim=512, n_blocks=8, schedule="cosine", dropout=0.1):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps

        self.net = NoisePredictor(
            geom_dim=geom_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
        )

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

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
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)
        # No CFG: condition is always provided, no masking
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.net(x_t, t, cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond, n_samples=128, **kwargs):
        """Sample without CFG: single forward pass per denoising step."""
        device = self.betas.device
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)

        x = torch.randn(n_samples, self.geom_dim, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            eps = self.net(x, t, cond_batch)

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
    model = ConditionalDDPM()
    x = torch.randn(4, 21).clamp(0, 1)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
