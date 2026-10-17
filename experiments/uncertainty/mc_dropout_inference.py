"""
MC Dropout 推理脚本
对测试集执行 T 次 dropout 激活的前向传播,得到预测均值 μ(x) 与方差 σ²(x)。

输出:
  mc_preds_test.npy : shape (T, N_test, 3)  反归一化预测
  test_labels.npy   : shape (N_test, 3)      反归一化真值
"""
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# Allow importing from parent project directory
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(HERE))  # ...\Performance_predict_transformer_012456
sys.path.append(PROJ_ROOT)

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.performance_predictor.vit import ViT
from models.performance_predictor.dataset import PerformanceDataset as input_dataset


T_FORWARD = 50           # MC Dropout 采样次数
SPLIT_SEED = 15          # 与 Test_Performance.py 一致
DATA_FILE = str(ROOT / "data/parameter_geometry_5200.npy")
LABEL_FILE = str(ROOT / "data/label_5200.npy")
CKPT_PATH = str(ROOT / "models/performance_predictor/checkpoints/best_model_in_testdataset.pth")


def build_model(device):
    return ViT(
        lenth=1, channels=21, num_classes=3,
        dim=256, depth=14, heads=12, mlp_dim=256,
        dropout=0.085, emb_dropout=0.072, dim_head=256
    ).to(device)


def enable_dropout(model):
    """Keep BN/LN in eval, but force Dropout into train mode for MC sampling."""
    n_active = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n_active += 1
    return n_active


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset + identical split to Test_Performance.py
    dataset = input_dataset(DATA_FILE, LABEL_FILE)
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    torch.manual_seed(SPLIT_SEED)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Val: {len(val_dataset)}")

    # Denormalization constants
    label_arr = np.load(LABEL_FILE)
    label_max = label_arr.max(axis=0)  # shape (3,)
    label_min = label_arr.min(axis=0)
    print(f"Label max: {label_max}, min: {label_min}")

    # Model
    model = build_model(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    n_drop = enable_dropout(model)
    print(f"Activated {n_drop} Dropout layers for MC sampling.")

    def run_mc(loader, split_name):
        """Collect labels + T-pass MC Dropout predictions for a given loader."""
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                all_labels.append(batch['label'].numpy())
        labels_norm = np.concatenate(all_labels, axis=0)
        labels_denorm = labels_norm * (label_max - label_min) + label_min  # (N, 3)

        N = labels_norm.shape[0]
        mc_preds = np.zeros((T_FORWARD, N, 3), dtype=np.float64)
        with torch.no_grad():
            for t in range(T_FORWARD):
                preds_t = []
                for batch in loader:
                    x = batch['data'].to(device)
                    p = model(x).cpu().numpy()
                    preds_t.append(p)
                preds_t = np.concatenate(preds_t, axis=0)
                mc_preds[t] = preds_t * (label_max - label_min) + label_min
                if (t + 1) % 10 == 0 or t == 0:
                    print(f"  [{split_name}] pass {t+1}/{T_FORWARD} done")

        mu = mc_preds.mean(axis=0)
        sigma = mc_preds.std(axis=0)
        print(f"\n--- {split_name} predictive stats ---")
        for i, name in enumerate(['Mass', 'Eff', 'Pre']):
            err = np.abs(mu[:, i] - labels_denorm[:, i])
            cov_2s = np.mean(err <= 2.0 * sigma[:, i])
            print(f"  {name:>4s}: MAE(mu)={err.mean():.5f}  mean sigma={sigma[:,i].mean():.5f}  Cov@2s={cov_2s*100:.1f}%")
        return mc_preds, labels_denorm

    # --- Train set (for consistent scatter / regression stats with MC mean) ---
    print("\n>>> Running MC Dropout on TRAIN set...")
    mc_train, labels_train = run_mc(train_loader, "TRAIN")
    np.save(os.path.join(HERE, 'mc_preds_train.npy'), mc_train)
    np.save(os.path.join(HERE, 'train_labels.npy'), labels_train)

    # --- Test set ---
    print("\n>>> Running MC Dropout on TEST set...")
    mc_test, labels_test = run_mc(test_loader, "TEST")
    np.save(os.path.join(HERE, 'mc_preds_test.npy'), mc_test)
    np.save(os.path.join(HERE, 'test_labels.npy'), labels_test)

    # --- Validation set (for temperature scaling calibration) ---
    print("\n>>> Running MC Dropout on VALIDATION set (for tau calibration)...")
    mc_val, labels_val = run_mc(val_loader, "VAL")
    np.save(os.path.join(HERE, 'mc_preds_val.npy'), mc_val)
    np.save(os.path.join(HERE, 'val_labels.npy'), labels_val)

    print(f"\nAll arrays saved to: {HERE}")


if __name__ == "__main__":
    main()
