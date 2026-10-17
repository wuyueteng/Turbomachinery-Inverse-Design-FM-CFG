"""
数据加载与预处理工具
=====================
统一的数据加载、归一化、划分等功能，供所有模型共享。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────── 路径配置 ──────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认使用 5200 数据集
GEOMETRY_PATH = os.path.join(ROOT_DIR, "data", "parameter_geometry_5200.npy")
LABEL_PATH = os.path.join(ROOT_DIR, "data", "label_5200.npy")
# 大数据集路径
MERGED_GEOMETRY_PATH = os.path.join(ROOT_DIR, "data", "merged_geometry.npy")
MERGED_LABEL_PATH = os.path.join(ROOT_DIR, "data", "merged_label.npy")

# 几何参数名称（21维）
GEOM_NAMES = [
    "Angle_in1", "Angle_out1", "Chord1", "THmax/CH1", "THmaxP1", "SWA1", "BOWA1",
    "Angle_in2", "Angle_out2", "Chord2", "THmax/CH2", "THmaxP2", "SWA2", "BOWA2",
    "Angle_in3", "Angle_out3", "Chord3", "THmax/CH3", "THmaxP3", "SWA3", "BOWA3",
]
PERF_NAMES = ["Flow (kg/s)", "Efficiency", "Pressure Ratio"]


# ────────────────────────────── 归一化统计 ──────────────────────────────
def load_norm_stats():
    """加载归一化统计量（基于 5200 数据集）"""
    geom_ref = np.load(GEOMETRY_PATH)
    label_ref = np.load(LABEL_PATH)

    stats = {
        "geom_min": geom_ref.min(axis=0),
        "geom_max": geom_ref.max(axis=0),
        "label_min": label_ref.min(axis=0),
        "label_max": label_ref.max(axis=0),
    }
    return stats


def normalize(data, dmin, dmax):
    return (data - dmin) / (dmax - dmin + 1e-12)


def denormalize(data, dmin, dmax):
    return data * (dmax - dmin) + dmin


# ────────────────────────────── Dataset ──────────────────────────────
class CompressorDataset(Dataset):
    """压气机几何-性能数据集"""
    def __init__(self, geometry, labels):
        self.geometry = torch.tensor(geometry, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.geometry)

    def __getitem__(self, idx):
        return {"geometry": self.geometry[idx], "label": self.labels[idx]}


# ────────────────────────────── DataLoader 工厂 ──────────────────────────────
def get_dataloaders(batch_size=256, train_ratio=0.8, val_ratio=0.1, seed=42,
                    use_merged=True):
    """
    加载数据并划分 train / val / test，返回 DataLoader 和归一化统计量。

    Args:
        use_merged: True 使用 merged 大数据集 (13840条), False 使用 5200 数据集.
                    归一化统计量始终基于 5200 数据集 (与 ViT 预测器一致).
    """
    if use_merged:
        geometry = np.load(MERGED_GEOMETRY_PATH)
        labels = np.load(MERGED_LABEL_PATH)
    else:
        geometry = np.load(GEOMETRY_PATH)
        labels = np.load(LABEL_PATH)

    # 归一化统计量始终基于 5200 数据集 (与 ViT 预测器一致)
    stats = load_norm_stats()

    geom_norm = normalize(geometry, stats["geom_min"], stats["geom_max"])
    label_norm = normalize(labels, stats["label_min"], stats["label_max"])

    N = len(geometry)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_set = CompressorDataset(geom_norm[train_idx], label_norm[train_idx])
    val_set = CompressorDataset(geom_norm[val_idx], label_norm[val_idx])
    test_set = CompressorDataset(geom_norm[test_idx], label_norm[test_idx])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    info = {
        "stats": stats,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_geom": geom_norm[train_idx],
        "train_label": label_norm[train_idx],
        "val_geom": geom_norm[val_idx],
        "val_label": label_norm[val_idx],
        "test_geom": geom_norm[test_idx],
        "test_label": label_norm[test_idx],
    }
    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    tr, va, te, info = get_dataloaders()
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
    for batch in tr:
        print("Geometry:", batch["geometry"].shape, "Label:", batch["label"].shape)
        break
