"""
Train DDPM-CFG (基于 DDPM 最佳超参 + CFG 消融)
================================================
直接使用 DDPM HPO 最佳架构超参, 只加 CFG 机制.
消融实验: 唯一变量 = CFG 有无.

运行:
  python models/ddpm_cfg/train.py
"""

import os, sys, time, copy, json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(os.path.dirname(MODEL_DIR))
sys.path.insert(0, ROOT_DIR)

from shared.data_utils import get_dataloaders
from models.ddpm_cfg.model import DDPM_CFG

CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
RESULT_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ddpm_cfg"


def train(args):
    print(f"[Device] {DEVICE}")
    train_loader, val_loader, _, info = get_dataloaders(batch_size=args.batch_size,
                                                         use_merged=True)
    print(f"Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")

    model = DDPM_CFG(
        geom_dim=21, cond_dim=3,
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        p_uncond=args.p_uncond,
        guidance_scale=args.guidance_scale,
    ).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            geom = batch["geometry"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            loss = model.training_loss(geom, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * geom.shape[0]
            n += geom.shape[0]
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_total, vn = 0, 0
            for batch in val_loader:
                geom = batch["geometry"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                loss = model.training_loss(geom, label)
                val_total += loss.item() * geom.shape[0]
                vn += geom.shape[0]
        val_loss = val_total / vn
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  best={best_val:.6f}  ({time.time()-t0:.1f}s)")

        if no_improve >= args.patience and epoch > 100:
            print(f"Early stop at epoch {epoch}")
            break

    ckpt_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}.pth")
    torch.save(best_state, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════════
    #  DDPM HPO best 架构 + CFG (p_uncond, guidance_scale)
    #  消融实验: 唯一变量 = CFG 有无
    # ════════════════════════════════════════════════════════════════
    import types
    args = types.SimpleNamespace(
        epochs         = 300,
        lr             = 9.78e-5,    # DDPM HPO best
        batch_size     = 256,
        patience       = 50,         # 与 DDPM HPO 一致
        timesteps      = 450,        # DDPM HPO best
        hidden_dim     = 1024,       # DDPM HPO best
        n_blocks       = 7,          # DDPM HPO best
        dropout        = 0.037,      # DDPM HPO best
        p_uncond       = 0.15,       # CFG 条件 dropout
        guidance_scale = 2.0,        # CFG 引导强度
    )
    train(args)
