"""
Train Conditional Flow Matching (无 CFG 消融版)
================================================
与 FM-CFG 训练流程相同, 区别:
  - 不做条件 dropout (p_uncond=0)
  - 无 guidance_scale 参数

运行:
  python models/fm/train.py
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
from models.fm.model import ConditionalFM

CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
RESULT_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "fm"


def train(args):
    print(f"[Device] {DEVICE}")
    train_loader, val_loader, _, info = get_dataloaders(batch_size=args.batch_size,
                                                         use_merged=True)
    print(f"Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")

    model = ConditionalFM(
        geom_dim=21, cond_dim=3,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        sampling_steps=args.sampling_steps,
        dropout=args.dropout,
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
    #  直接使用 FM-CFG 的 HPO 最佳超参, 只去掉 CFG
    #  (消融实验: 唯一变量 = CFG 有无)
    # ════════════════════════════════════════════════════════════════
    import types
    args = types.SimpleNamespace(
        epochs        = 300,
        lr            = 0.000454,   # FM-CFG HPO best
        batch_size    = 256,
        patience      = 50,         # 与 FM-CFG HPO 一致
        hidden_dim    = 512,        # FM-CFG HPO best
        n_blocks      = 5,          # FM-CFG HPO best
        sampling_steps = 100,       # FM-CFG HPO best
        dropout       = 0.081,      # FM-CFG HPO best
    )
    train(args)
