import torch as th
from torch.utils.data import Dataset

from minerl3161.buffers import ReplayBuffer

class MineRLDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
