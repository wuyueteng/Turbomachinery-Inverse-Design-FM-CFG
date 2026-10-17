"""
训练 ViT 性能预测模型
=====================
用法:
    python train.py

数据: 需要准备 parameter_geometry_5200.npy 和 label_5200.npy
"""

import os, sys, argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vit import ViT
from dataset import PerformanceDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_data(data_file, label_file, batch_size=32, seed=15):
    dataset = PerformanceDataset(data_file, label_file)
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    validation_size = len(dataset) - train_size - test_size
    torch.manual_seed(seed)
    train_dataset, test_dataset, validation_dataset = random_split(
        dataset, [train_size, test_size, validation_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def build_model():
    return ViT(
        lenth=1, channels=21, num_classes=3,
        dim=256, depth=14, heads=12, mlp_dim=256,
        dropout=0.085, emb_dropout=0.072, dim_head=256,
    ).to(DEVICE)


def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    losses = []
    for batch in tqdm(dataloader):
        x, y = batch['data'].to(DEVICE), batch['label'].to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def validate(model, dataloader, loss_fn):
    model.eval()
    losses, abs_errors, rel_errors = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['data'].to(DEVICE), batch['label'].to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            abs_err = torch.abs(pred - y).mean().item()
            rel_err = (torch.abs(pred - y) / (torch.abs(y) + 1e-8)).mean().item()
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
    return (sum(losses) / len(losses),
            sum(abs_errors) / len(abs_errors),
            sum(rel_errors) / len(rel_errors))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the Transformer performance predictor")
    parser.add_argument('--data', default=str(root/'data/parameter_geometry_5200.npy'))
    parser.add_argument('--labels', default=str(root/'data/label_5200.npy'))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--split-seed', type=int, default=15)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent/'checkpoints'))
    args = parser.parse_args()
    train_loader, test_loader = load_data(args.data, args.labels, args.batch_size, args.split_seed)

    model = build_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_epochs = args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_test_loss = float("inf")
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        scheduler.step()
        test_loss, avg_abs, avg_rel = validate(model, test_loader, loss_fn)
        model.train()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_in_testdataset.pth"))
            print(f"Saved best model (test loss: {best_test_loss:.5f})")

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{n_epochs}], Train: {train_loss:.5f}, '
                  f'Test: {test_loss:.5f}, AbsErr: {avg_abs:.5f}, RelErr: {avg_rel:.5f}')

    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
