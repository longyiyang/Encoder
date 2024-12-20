from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, data, seq_len=20):
        # 数据处理：填充或截断到固定长度
        self.data = [d[:seq_len] if len(d) >= seq_len else d + [0] * (seq_len - len(d)) for d in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回一个样本
        return torch.tensor(self.data[idx])


