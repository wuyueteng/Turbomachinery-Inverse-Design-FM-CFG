"""
Conditional GAN (cGAN) with WGAN-GP
=====================================
条件生成对抗网络，采用 Wasserstein 距离 + 梯度惩罚。
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
            nn.Linear(dim, dim * 4), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class Generator(nn.Module):
    def __init__(self, noise_dim=64, cond_dim=3, geom_dim=21,
                 hidden_dim=512, n_blocks=6):
        super().__init__()
        self.input_proj = nn.Linear(noise_dim + cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, geom_dim), nn.Sigmoid(),
        )

    def forward(self, z, cond):
        h = self.input_proj(torch.cat([z, cond], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class Discriminator(nn.Module):
    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512, n_blocks=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geom_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2), nn.Dropout(0.1),
            ) for _ in range(n_blocks)],
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, cond):
        return self.net(torch.cat([x, cond], dim=-1))


class ConditionalGAN(nn.Module):
    """WGAN-GP 条件 GAN"""
    def __init__(self, geom_dim=21, cond_dim=3, noise_dim=64,
                 hidden_dim=512, n_g_blocks=6, n_d_blocks=4,
                 n_critic=5, lambda_gp=10.0, lr_g=1e-4, lr_d=1e-4):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.generator = Generator(noise_dim, cond_dim, geom_dim, hidden_dim, n_g_blocks)
        self.discriminator = Discriminator(geom_dim, cond_dim, hidden_dim, n_d_blocks)
        self.opt_g = None
        self.opt_d = None
        self._step_count = 0

    def _gradient_penalty(self, real, fake, cond):
        B = real.shape[0]
        alpha = torch.rand(B, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp, cond)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def training_loss(self, x_real, cond):
        device = x_real.device
        B = x_real.shape[0]
        if self.opt_g is None:
            self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
            self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        z = torch.randn(B, self.noise_dim, device=device)
        fake = self.generator(z, cond)

        if self._step_count % (self.n_critic + 1) < self.n_critic:
            d_real = self.discriminator(x_real, cond).mean()
            d_fake = self.discriminator(fake.detach(), cond).mean()
            gp = self._gradient_penalty(x_real, fake.detach(), cond)
            d_loss = -d_real + d_fake + self.lambda_gp * gp
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            loss = d_loss
        else:
            g_loss = -self.discriminator(fake, cond).mean()
            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
            loss = g_loss

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, cond, n_samples=10, **kwargs):
        device = next(self.parameters()).device
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)
        z = torch.randn(n_samples, self.noise_dim, device=device)
        return self.generator(z, cond_batch).clamp(0, 1).cpu().numpy()

    @torch.no_grad()
    def sample_unconditional(self, n_samples=256):
        device = next(self.parameters()).device
        cond = torch.rand(n_samples, self.cond_dim, device=device)
        z = torch.randn(n_samples, self.noise_dim, device=device)
        return self.generator(z, cond).clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = ConditionalGAN()
    x = torch.rand(8, 21)
    c = torch.rand(8, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}")
