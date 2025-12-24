import torch
from torch.utils.data import Dataset
import random

# class MyDataset(Dataset):
#     def __init__(self, data, block_size):
#         self.data = data
#         self.block_size = block_size

#     def __len__(self):
#        return len(self.data)
    
#     def __getitem__(self, idx):
#         ix = random.randint(0, len(self.data) - self.block_size)
#         x = self.data[ix:ix + self.block_size]
#         y = self.data[ix + 1: ix + self.block_size + 1]
#         return x, y
    
class MyDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        self.data_recur = []
        for i in range(len(self.data) - self.block_size):
            # ix = random.randint(0, len(self.data) - self.block_size)
            x = self.data[i:i + self.block_size]
            y = self.data[i + 1: i + self.block_size + 1]
            self.data_recur.append((x, y))
        
    def __len__(self):
       return len(self.data_recur)
    
    def __getitem__(self, idx):
        return self.data_recur[idx]