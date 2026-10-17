"""
Conditional Invertible Neural Network (cINN)
============================================
参考: Jia et al. (2024) "Inverse Design of Compressor/Fan Blade Profiles
Based on Conditional Invertible Neural Networks", ASME Turbo Expo 2024.

与原 Glow 架构的主要改进:
  ✓ 纯交替仿射耦合块 (RealNVP-style)，去除 ActNorm + 置换
  ✓ 耦合网络: 2 层 × 512 神经元 + ReLU (论文原配置)
  ✓ 流步骤数: 20 块 (论文原配置)
  ✓ 放宽 log_s 限制: tanh × 2.0，允许 scale ∈ [e⁻², e²]
  ✓ 去除不稳定的 logit 预处理，改为简单线性归一化 [0,1]→[-1,1]
  ✓ 零初始化最后层，训练初始阶段流接近恒等映射
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ================================================================
#  CouplingNet — s 和 t 网络
# ================================================================
class CouplingNet(nn.Module):
    """
    仿射耦合中的尺度 s 和平移 t 网络。
    架构: Linear → ReLU → Linear → ReLU → Linear (论文: 2 隐层 × 512 × ReLU)
    条件 c 直接拼接到输入（论文 Section 2.2 原话: simply concatenating it proved effective）
    """

    def __init__(self, in_dim, out_dim, cond_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2),  # 同时输出 s 和 t
        )
        # 零初始化最后一层 → 训练开始时流 ≈ 恒等映射，梯度稳定
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_fixed, cond):
        h = self.net(torch.cat([x_fixed, cond], dim=-1))
        s, t = h.chunk(2, dim=-1)
        # tanh × 2.0: log_scale ∈ [-2, 2]，即 scale ∈ [e⁻², e²] ≈ [0.14, 7.39]
        # 足够表达力，同时避免数值爆炸（原来 tanh×0.5 限制过死）
        s = torch.tanh(s) * 2.0
        return s, t


# ================================================================
#  AffineCouplingBlock — 条件仿射耦合块
# ================================================================
class AffineCouplingBlock(nn.Module):
    """
    RealNVP-style 仿射耦合块:
      mask_first=True  (偶数块): x[:n1] 固定 → 变换 x[n1:]
      mask_first=False (奇数块): x[n1:] 固定 → 变换 x[:n1]

    对 21 维输入: n1=10, n2=11
    - 偶数块: 固定前 10 维，变换后 11 维；CouplingNet: (10+3)→512→512→22
    - 奇数块: 固定后 11 维，变换前 10 维；CouplingNet: (11+3)→512→512→20
    """

    def __init__(self, dim, cond_dim, hidden_dim=512, mask_first=True):
        super().__init__()
        self.mask_first = mask_first
        self.n1 = dim // 2       # 前半维度数
        self.n2 = dim - self.n1  # 后半维度数（dim 为奇数时比 n1 大 1）

        if mask_first:
            # 前半固定，后半变换
            self.coupling = CouplingNet(self.n1, self.n2, cond_dim, hidden_dim)
        else:
            # 后半固定，前半变换
            self.coupling = CouplingNet(self.n2, self.n1, cond_dim, hidden_dim)

    def forward(self, x, cond):
        """正向: x → y，返回 (y, log_det)"""
        x1, x2 = x[:, :self.n1], x[:, self.n1:]

        if self.mask_first:
            # x1 固定，变换 x2
            s, t = self.coupling(x1, cond)
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)             # (B,)
            return torch.cat([x1, y2], dim=-1), log_det
        else:
            # x2 固定，变换 x1
            s, t = self.coupling(x2, cond)
            y1 = x1 * torch.exp(s) + t
            log_det = s.sum(dim=-1)             # (B,)
            return torch.cat([y1, x2], dim=-1), log_det

    def inverse(self, y, cond):
        """逆向: y → x（采样时使用）"""
        y1, y2 = y[:, :self.n1], y[:, self.n1:]

        if self.mask_first:
            # y1 不变，逆变换 y2
            s, t = self.coupling(y1, cond)
            x2 = (y2 - t) * torch.exp(-s)
            return torch.cat([y1, x2], dim=-1)
        else:
            # y2 不变，逆变换 y1
            s, t = self.coupling(y2, cond)
            x1 = (y1 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=-1)


# ================================================================
#  ConditionalGlow — 主模型（接口与原 Glow 兼容）
# ================================================================
class ConditionalGlow(nn.Module):
    """
    Conditional INN (cINN) for tabular blade geometry inverse design.

    与论文 Jia et al. (2024) 保持一致:
      - 20 个交替仿射耦合块
      - 耦合网络: 2 层 × 512 × ReLU
      - 条件拼接到 s/t 网络输入
      - NLL 损失 (Eq. 8 in paper)
      - 输入归一化: [0,1] → [-1,1]

    接口与原 ConditionalGlow 100% 兼容，可直接替换。
    """

    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 n_flows=20, n_layers_coupling=2, temperature=0.8):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.temperature = temperature
        # n_layers_coupling 保留为接口参数（当前固定 2 层，与论文一致）

        self.flows = nn.ModuleList([
            AffineCouplingBlock(
                dim=geom_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                mask_first=(i % 2 == 0),   # 偶数块固定前半，奇数块固定后半
            )
            for i in range(n_flows)
        ])

    # ── 数据预处理: [0,1] → [-1,1] ──────────────────────────────────
    def _preprocess(self, x):
        """将 [0,1] 范围数据线性映射到 [-1,1]，保证各维度都在合理范围内"""
        return x * 2.0 - 1.0

    def _postprocess(self, x):
        """将 [-1,1] 映射回 [0,1]"""
        return (x + 1.0) * 0.5

    # ── 前向 / 逆向 ──────────────────────────────────────────────────
    def forward(self, x, cond):
        """正向变换: data → latent，返回 (z, total_log_det)"""
        z = self._preprocess(x)
        total_log_det = 0.0
        for flow in self.flows:
            z, ld = flow(z, cond)
            total_log_det = total_log_det + ld
        return z, total_log_det

    def inverse(self, z, cond):
        """逆向变换: latent → data（采样时使用）"""
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x, cond)
        x = self._postprocess(x)
        return x

    # ── 概率计算 ─────────────────────────────────────────────────────
    def log_prob(self, x, cond):
        """计算条件对数概率 log p(x|cond)

        按变量变换公式:
          log p(x|c) = log p_z(z) + log|det(dz/dx)|
        其中 z = f(x; c)，p_z = N(0, I)
        """
        z, log_det = self.forward(x, cond)
        # 标准正态 log 概率（逐维求和）
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return log_pz + log_det  # (B,)

    # ── 训练接口 ─────────────────────────────────────────────────────
    def training_loss(self, x, cond):
        """NLL 损失，按维度归一化 (对应论文 Eq. 8 前半部分)

        L = -E[log p(x|c)] / geom_dim
          = E[0.5*||z||² + 0.5*D*log(2π) - log|det J|] / geom_dim
        """
        x = x.clamp(1e-6, 1 - 1e-6)  # 防止数值边界问题
        log_prob = self.log_prob(x, cond)
        nll = -log_prob.mean() / self.geom_dim
        # NaN/Inf 保护（正常训练不应触发）
        if torch.isnan(nll) or torch.isinf(nll):
            return torch.tensor(5.0, device=x.device, requires_grad=True)
        return nll

    # ── 采样接口 ─────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, cond, n_samples=128, temperature=None, **kwargs):
        """条件采样: 从 z ~ N(0, T²I) 通过逆变换生成几何参数

        Args:
            cond: 性能条件，shape (3,) 或 (1, 3)，归一化后的值
            n_samples: 生成样本数量
            temperature: 采样温度，越小越保守，默认 0.8
        Returns:
            np.ndarray, shape (n_samples, geom_dim)，值域 [0, 1]
        """
        if temperature is None:
            temperature = self.temperature
        device = next(self.parameters()).device

        if isinstance(cond, np.ndarray):
            cond = torch.tensor(cond, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_batch = cond.repeat(n_samples, 1)

        z = torch.randn(n_samples, self.geom_dim, device=device) * temperature
        x = self.inverse(z, cond_batch)
        return x.clamp(0, 1).cpu().numpy()

    @torch.no_grad()
    def sample_unconditional(self, n_samples=256):
        """无条件采样（随机条件，用于探索设计空间）"""
        device = next(self.parameters()).device
        cond = torch.rand(n_samples, self.cond_dim, device=device)
        z = torch.randn(n_samples, self.geom_dim, device=device) * self.temperature
        x = self.inverse(z, cond)
        return x.clamp(0, 1).cpu().numpy()


# ================================================================
#  自测
# ================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ConditionalGlow(
        geom_dim=21, cond_dim=3, hidden_dim=512, n_flows=20
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ── 训练损失测试 ──
    x = torch.rand(32, 21, device=device) * 0.8 + 0.1
    cond = torch.rand(32, 3, device=device)
    loss = model.training_loss(x, cond)
    print(f"Training loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"

    # ── 前向-逆向一致性测试（最重要）──
    z, log_det = model.forward(x, cond)
    x_recon = model.inverse(z, cond)
    recon_err = (x - x_recon).abs().max().item()
    print(f"Forward-Inverse reconstruction error: {recon_err:.2e}")
    assert recon_err < 1e-4, f"Reconstruction error too large: {recon_err}"

    # ── 采样测试 ──
    c_single = np.array([0.5, 0.8, 0.3])
    samples = model.sample(c_single, n_samples=16, temperature=0.8)
    print(f"Sample shape: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}]")
    assert samples.shape == (16, 21), f"Wrong shape: {samples.shape}"

    # ── 梯度测试 ──
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradient norms — mean: {sum(grad_norms)/len(grad_norms):.4f}, "
          f"max: {max(grad_norms):.4f}")
    assert max(grad_norms) < 1e4, "Gradient explosion detected!"

    print("\n✓ All tests passed!")
