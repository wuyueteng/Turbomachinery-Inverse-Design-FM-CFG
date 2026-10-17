"""
Conditional Variational Autoencoder (cVAE)
============================================
Encoder: 几何参数 + 性能条件 → 隐空间 μ, σ
Decoder: 隐变量 z + 性能条件 → 几何参数
损失: 重构损失 + β·KL 散度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 latent_dim=32, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(geom_dim + cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, cond):
        h = self.input_proj(torch.cat([x, cond], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.mu_proj(h), self.logvar_proj(h)


class Decoder(nn.Module):
    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 latent_dim=32, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, geom_dim), nn.Sigmoid(),
        )

    def forward(self, z, cond):
        h = self.input_proj(torch.cat([z, cond], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class ConditionalVAE(nn.Module):
    """条件变分自编码器"""
    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 latent_dim=32, n_layers=4, beta=1.0, dropout=0.1):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(geom_dim, cond_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = Decoder(geom_dim, cond_dim, hidden_dim, latent_dim, n_layers, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, cond), mu, logvar

    def training_loss(self, x, cond):
        recon, mu, logvar = self.forward(x, cond)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    @torch.no_grad()
    def sample(self, cond, n_samples=128, **kwargs):
        device = next(self.parameters()).device
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z, cond_batch).clamp(0, 1).cpu().numpy()

    @torch.no_grad()
    def sample_unconditional(self, n_samples=256):
        device = next(self.parameters()).device
        cond = torch.rand(n_samples, self.cond_dim, device=device)
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z, cond).clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = ConditionalVAE()
    x = torch.randn(4, 21).clamp(0, 1)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
