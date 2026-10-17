"""
PA-FM-CFG: Performance-Aware Flow Matching with Classifier-Free Guidance
==========================================================================
在 FM-CFG 基础上引入代理性能损失 (surrogate performance loss).
通过少步 ODE 从中间状态 x_t 估算最终设计 x_1_hat, 用冻结 ViT 预测其性能,
与目标条件计算 MSE 作为额外损失信号.

训练损失:
  L_total = L_flow + lambda(epoch) * L_perf(x_1_hat)

关键设计:
  - 少步 ODE 估计 (ode_steps=K): 从 x_t 走 K 步到 t=1
  - 前 K-1 步 no_grad (省显存), 最后 1 步保留梯度 (反传到速度场网络)
  - t 过滤: 只对 t > t_min 的样本计算 perf loss (小 t 时估计不准)
  - t 加权: t^1.5 权重, 后期样本贡献更大
  - lambda 调度: warmup (纯 flow loss) -> 线性 ramp -> 最终 lambda_perf

推理采样与 FM-CFG 完全相同 (ODE + CFG).

HPO 搜索参数 (FM-CFG 架构固定, 只搜 PA 相关):
  lambda_perf, ode_steps, t_min, warmup_epochs, ramp_epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.fm_cfg.model import VelocityNet
from models.pa_fm_cfg.surrogate_guide import SurrogatePerformanceGuide


class PerformanceAwareFlowMatching(nn.Module):
    """PA-FM-CFG: FM-CFG + surrogate performance loss with few-step ODE estimation."""

    def __init__(self, geom_dim=21, cond_dim=3, hidden_dim=512,
                 n_blocks=8, p_uncond=0.15, guidance_scale=2.0,
                 sampling_steps=100, dropout=0.1,
                 lambda_perf=0.2, warmup_epochs=80, ramp_epochs=50,
                 ode_steps=5, t_min=0.2,
                 perf_weights=None, device="cuda"):
        super().__init__()
        self.geom_dim = geom_dim
        self.cond_dim = cond_dim
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.sampling_steps = sampling_steps
        self.lambda_perf = lambda_perf
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.ode_steps = ode_steps
        self.t_min = t_min
        self.perf_weights = perf_weights

        self.net = VelocityNet(
            geom_dim=geom_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
        )
        self.surrogate = SurrogatePerformanceGuide(device=device)

    def _get_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            return self.lambda_perf * (epoch - self.warmup_epochs) / self.ramp_epochs
        return self.lambda_perf

    def _few_step_ode_estimate(self, x_start, t_start, cond, n_steps):
        """
        Few-step ODE from x_t at t_start to t=1.
        First K-1 steps: no_grad. Last step: retains gradient.
        """
        dt = ((1.0 - t_start) / n_steps).unsqueeze(-1)
        x = x_start.detach()
        t_cur = t_start.clone()
        with torch.no_grad():
            for step in range(n_steps - 1):
                v = self.net(x, t_cur, cond)
                x = x + v * dt
                t_cur = t_cur + dt.squeeze(-1)
        v_final = self.net(x, t_cur, cond)
        return (x + v_final * dt).clamp(0, 1)

    def training_loss(self, x_1, cond, epoch=0):
        B = x_1.shape[0]
        device = x_1.device

        # Flow matching loss (with CFG masking)
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        target = x_1 - x_0

        mask = torch.rand(B, device=device) < self.p_uncond
        cond_masked = cond.clone()
        cond_masked[mask] = 0.0

        v_pred = self.net(x_t, t, cond_masked)
        loss_flow = F.mse_loss(v_pred, target)

        # Surrogate performance loss
        lam = self._get_lambda(epoch)
        if lam > 0:
            perf_mask = t > self.t_min
            if perf_mask.sum() > 0:
                x_1_hat = self._few_step_ode_estimate(
                    x_t[perf_mask], t[perf_mask], cond[perf_mask], self.ode_steps)
                pred_perf = self.surrogate.predict(x_1_hat)
                cond_sel = cond[perf_mask]
                if self.perf_weights is not None:
                    w = torch.tensor(self.perf_weights, device=device, dtype=pred_perf.dtype)
                    per_sample = (w * (pred_perf - cond_sel) ** 2).mean(dim=1)
                else:
                    per_sample = ((pred_perf - cond_sel) ** 2).mean(dim=1)
                t_weight = (t[perf_mask] ** 1.5).detach()
                loss_perf = (t_weight * per_sample).mean()
            else:
                loss_perf = torch.tensor(0.0, device=device)
        else:
            loss_perf = torch.tensor(0.0, device=device)

        total_loss = loss_flow + lam * loss_perf
        return total_loss, {"flow": loss_flow.item(), "perf": loss_perf.item(), "lambda": lam}

    @torch.no_grad()
    def sample(self, cond, n_samples=128, guidance_scale=None, steps=None, **kwargs):
        """ODE sampling with CFG (identical to FM-CFG)."""
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
            t_val = torch.full((n_samples,), i * dt, device=device)
            v_cond = self.net(x, t_val, cond_batch)
            v_uncond = self.net(x, t_val, null_cond)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + v * dt
        return x.clamp(0, 1).cpu().numpy()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
