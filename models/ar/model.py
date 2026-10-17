"""
Conditional Autoregressive Transformer (AR)
============================================
自回归生成模型：逐参数生成 21 维几何参数。
每个参数的分布由前面已生成的参数和性能条件共同决定。

架构：
- 条件 token (性能条件) + 21 个参数 token → Causal Transformer
- 每个位置输出 (mu, log_var)，采样时使用温度缩放
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalAutoregressive(nn.Module):
    """条件自回归 Transformer"""

    def __init__(self, geom_dim=21, cond_dim=3, dim=256,
                 n_heads=4, n_layers=6, dropout=0.1, temperature=0.8):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.dim = dim
        self.temperature = temperature

        seq_len = geom_dim + 1  # COND token + 21 param tokens

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.value_embed = nn.Linear(1, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
        )

        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x, cond):
        """
        Teacher-forced forward.
        x:    (B, 21)
        cond: (B, 3)
        返回: (B, 21, 2) 每个位置的 (mu, log_var)
        """
        cond_token = self.cond_embed(cond).unsqueeze(1)
        param_tokens = self.value_embed(x.unsqueeze(-1))
        tokens = torch.cat([cond_token, param_tokens], dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, self.causal_mask)

        out = self.output_head(tokens[:, :self.geom_dim, :])
        return out

    def training_loss(self, x, cond):
        """NLL 训练损失"""
        out = self.forward(x, cond)
        mu = out[:, :, 0]
        log_var = out[:, :, 1].clamp(-10, 2)
        nll = 0.5 * (log_var + (x - mu) ** 2 / (log_var.exp() + 1e-8))
        return nll.mean()

    @torch.no_grad()
    def sample(self, cond, n_samples=10, temperature=None, **kwargs):
        """自回归采样，支持温度控制"""
        if temperature is None:
            temperature = self.temperature

        device = next(self.parameters()).device
        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)

        x_gen = torch.zeros(n_samples, self.geom_dim, device=device)

        for i in range(self.geom_dim):
            out = self.forward(x_gen, cond_batch)
            mu = out[:, i, 0]
            log_var = out[:, i, 1].clamp(-10, 2)
            std = (0.5 * log_var).exp() * temperature
            x_gen[:, i] = (mu + std * torch.randn_like(std)).clamp(0, 1)

        return x_gen.cpu().numpy()

    @torch.no_grad()
    def sample_unconditional(self, n_samples=256):
        device = next(self.parameters()).device
        cond = torch.rand(n_samples, self.cond_dim, device=device)
        return np.vstack([self.sample(cond[i], n_samples=1) for i in range(n_samples)])


if __name__ == "__main__":
    model = ConditionalAutoregressive()
    x = torch.rand(4, 21)
    c = torch.rand(4, 3)
    loss = model.training_loss(x, c)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    samples = model.sample(c[0], n_samples=5)
    print(f"Samples: {samples.shape}")
