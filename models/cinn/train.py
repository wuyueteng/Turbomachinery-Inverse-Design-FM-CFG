"""
cINN (Conditional Glow) Training Script
========================================
训练配置参考 Jia et al. (2024):
  - Adam: lr=0.001
  - LR 调度: epoch 30 → 0.0005, epoch 40 → 0.00001
  - 100 epochs
  - NLL 损失（按维度归一化）

用法:
    python models/glow/train.py
    python models/glow/train.py --epochs 100 --batch_size 512 --hidden_dim 512
"""

import os
import sys
import time

import torch
import torch.optim as optim

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(os.path.dirname(MODEL_DIR))
sys.path.insert(0, ROOT_DIR)

from shared.data_utils import get_dataloaders
from models.glow.model import ConditionalGlow

CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
RESULT_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "cinn"


def parse_args():
    # ════════════════════════════════════════════════════════════════
    #  直接修改下面的参数，然后运行: python models/cinn/train.py
    # ════════════════════════════════════════════════════════════════
    import types
    return types.SimpleNamespace(
        epochs       = 100,    # 训练轮数
        batch_size   = 512,    # 批大小
        hidden_dim   = 512,    # 耦合层隐层维度
        n_flows      = 20,     # 归一化流层数
        lr           = 1e-3,   # 学习率
        temperature  = 0.8,    # 采样温度（推理阶段）
        weight_decay = 1e-5,   # L2 正则化系数
        patience     = 20,     # 早停耐心值
        seed         = 42,     # 随机种子
        force        = False,  # True → 强制重新训练（忽略已有权重）
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.n_flows} flows, hidden={args.hidden_dim}, "
          f"lr={args.lr}, bs={args.batch_size}, epochs={args.epochs}")

    # ── 数据 ──────────────────────────────────────────────────────────
    train_loader, val_loader, _, info = get_dataloaders(
        batch_size=args.batch_size, seed=args.seed
    )
    print(f"Train: {info['n_train']} | Val: {info['n_val']} | "
          f"Geom dim: {info['geom_dim']} | Cond dim: {info['label_dim']}")

    # ── 模型 ──────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, "Glow.pth")

    model = ConditionalGlow(
        geom_dim=info["geom_dim"],
        cond_dim=info["label_dim"],
        hidden_dim=args.hidden_dim,
        n_flows=args.n_flows,
        temperature=args.temperature,
    ).to(device)

    if not args.force and os.path.exists(ckpt_path):
        print(f"[Skip] Checkpoint already exists: {ckpt_path}")
        print("  Use --force to retrain.")
        return

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── 优化器 & 调度器 ───────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 论文训练配置: lr: 0.001 → 0.0005 (epoch 30) → 0.00001 (epoch 40)
    def lr_lambda(epoch):
        if epoch < 30:
            return 1.0
        elif epoch < 40:
            return 0.5      # 0.001 * 0.5 = 0.0005
        else:
            return 0.01     # 0.001 * 0.01 = 0.00001

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 训练循环 ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    print(f"\n{'Epoch':>6} {'Train NLL':>12} {'Val NLL':>12} {'LR':>10} {'Time':>8}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── 训练 ──
        model.train()
        train_losses = []
        for x_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)

            optimizer.zero_grad()
            loss = model.training_loss(x_batch, c_batch)
            loss.backward()
            # 梯度裁剪，防止训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── 验证 ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                loss = model.training_loss(x_batch, c_batch)
                val_losses.append(loss.item())

        train_nll = sum(train_losses) / len(train_losses)
        val_nll   = sum(val_losses)   / len(val_losses)
        cur_lr    = optimizer.param_groups[0]["lr"]
        elapsed   = time.time() - t0

        history["train"].append(train_nll)
        history["val"].append(val_nll)

        print(f"{epoch:>6} {train_nll:>12.4f} {val_nll:>12.4f} "
              f"{cur_lr:>10.2e} {elapsed:>7.1f}s")

        # ── Best checkpoint ──
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_nll,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [OK] Saved best model (val NLL={val_nll:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nTraining complete. Best val NLL: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")

    # ── 保存训练曲线 ──
    try:
        import json
        history_path = os.path.join(RESULT_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history: {history_path}")
    except Exception as e:
        print(f"Warning: Could not save history: {e}")

    # ── 快速验证：前向-逆向一致性 ──
    print("\n[Validation] Forward-Inverse consistency check...")
    model.eval()
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["model_state_dict"]
    )
    x_test, c_test = next(iter(val_loader))
    x_test = x_test[:8].to(device)
    c_test  = c_test[:8].to(device)
    with torch.no_grad():
        z, _ = model.forward(x_test, c_test)
        x_recon = model.inverse(z, c_test)
    err = (x_test - x_recon).abs().max().item()
    status = "OK" if err < 1e-3 else "WARNING: error too large"
    print(f"  Max reconstruction error: {err:.2e}  [{status}]")

    # ── 采样演示 ──
    print("[Validation] Conditional sampling test...")
    import numpy as np
    c_demo = np.array([0.6, 0.7, 0.5])
    with torch.no_grad():
        samples = model.sample(c_demo, n_samples=8, temperature=args.temperature)
    print(f"  Sample shape: {samples.shape}, "
          f"range: [{samples.min():.3f}, {samples.max():.3f}]")
    print("Done.")


if __name__ == "__main__":
    main()
