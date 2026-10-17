"""性能预测模型 Dataset"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PerformanceDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file)
        self.label = np.load(label_file)
        self.max = self.data.max(axis=0)
        self.min = self.data.min(axis=0)
        self.data = (self.data - self.min) / (self.max - self.min)
        self.max1 = self.label.max(axis=0)
        self.min1 = self.label.min(axis=0)
        self.label = (self.label - self.min1) / (self.max1 - self.min1)
        b, l = self.data.shape
        self.data = self.data.reshape(b, 21, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
        }
