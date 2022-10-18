from typing import Dict

import torch as th
from torch.utils.data import Dataset
import numpy as np

from minerl3161.buffers import ReplayBuffer


class MineRLDataset(Dataset):
    """
    Dataset wrapper for PyTorch Lightning
    """

    def __init__(self, buffer: ReplayBuffer) -> None:
        """
        Initialiser for MineRLDataset

        Args:
            buffer (ReplayBuffer): the buffer that is being used as the dataset for pretraining
        """
        self.buffer = buffer

    def __len__(self) -> int:
        """
        Gets the length of the buffer

        Returns:
            int: length of buffer
        """
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Gets the item at index idx in the buffer

        Args:
            idx (int): the index of the item being retrieved
        
        Returns:
            Dict[str, np.ndarray]: the item being retrieved
        """
        return self.buffer[idx]
