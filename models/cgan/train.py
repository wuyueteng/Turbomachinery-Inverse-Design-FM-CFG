"""
Train Conditional GAN (WGAN-GP)
=================================
注意: GAN 使用内置双优化器，无需外部 optimizer。
运行:
  python models/cgan/train.py
  python models/cgan/train.py --epochs 500
"""

import os, sys, time, json
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(os.path.dirname(MODEL_DIR))
sys.path.insert(0, ROOT_DIR)

from shared.data_utils import get_dataloaders
from models.cgan.model import ConditionalGAN

CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
RESULT_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "cgan"


def train(args):
    print(f"[Device] {DEVICE}")
    train_loader, _, _, info = get_dataloaders(batch_size=args.batch_size)
    print(f"Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")

    model = ConditionalGAN(
        geom_dim=21, cond_dim=3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
    ).to(DEVICE)
    print(f"Generator params:     {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in model.discriminator.parameters()):,}")

    history = {"train": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for batch in train_loader:
            geom = batch["geometry"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            loss = model.training_loss(geom, label)
            epoch_losses.append(loss.item())
        avg = sum(epoch_losses) / len(epoch_losses)
        history["train"].append(avg)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d}/{args.epochs}  loss={avg:.6f}  ({time.time()-t0:.1f}s)")

    ckpt_path = os.path.join(CKPT_DIR, f"{MODEL_NAME}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved -> {ckpt_path}")

    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════════
    #  直接修改下面的参数，然后运行: python models/cgan/train.py
    # ════════════════════════════════════════════════════════════════
    import types
    args = types.SimpleNamespace(
        epochs     = 300,    # 训练轮数
        batch_size = 256,    # 批大小
        hidden_dim = 256,    # 生成器/判别器隐层维度
        latent_dim = 32,     # 噪声向量维度
        lr_g       = 1e-4,   # 生成器学习率
        lr_d       = 1e-4,   # 判别器学习率
        n_critic   = 5,      # 每次更新生成器前训练判别器的步数
        lambda_gp  = 10.0,   # 梯度惩罚系数
    )
    train(args)
