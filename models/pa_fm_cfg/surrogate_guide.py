"""
Surrogate Performance Guide (frozen ViT)
==========================================
Wraps the pre-trained ViT performance predictor as a differentiable surrogate loss.
Unlike shared/evaluation.py, gradients flow through the input geometry (ViT weights frozen).

Used by PA-FM-CFG to provide performance-aware training signal.
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "performance_predictor"))
from vit import ViT


class SurrogatePerformanceGuide(nn.Module):
    """Frozen ViT surrogate for differentiable performance prediction during training."""

    PREDICTOR_CKPT = os.path.join(
        ROOT_DIR, "models", "performance_predictor", "checkpoints", "best_model_in_testdataset.pth"
    )

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.vit = ViT(
            lenth=1, channels=21, num_classes=3,
            dim=256, depth=14, heads=12, mlp_dim=256,
            dropout=0.085, emb_dropout=0.072, dim_head=256,
        )
        checkpoint = torch.load(self.PREDICTOR_CKPT, map_location=device)
        self.vit.load_state_dict(checkpoint)
        self.vit.requires_grad_(False)
        self.vit.eval()
        self.vit.to(device)
        print(f"[SurrogateGuide] ViT loaded (frozen): {self.PREDICTOR_CKPT}")

    def predict(self, geometry_norm: torch.Tensor) -> torch.Tensor:
        """Differentiable performance prediction (no torch.no_grad)."""
        return self.vit(geometry_norm.unsqueeze(2))

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self
