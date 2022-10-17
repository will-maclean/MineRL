from typing import Dict

import torch as th
from torch import nn
import numpy as np


class TinyDQN(nn.Module):
    """
    Stores the PyTorch neural network to be used as a TinyDQN network.
    """

    def __init__(self, S: int, A: int) -> None:
        """
        Initialiser for TinyDQN

        Args:
            S (int): represents the number of input values that corresponds to the number of neurons in the input layer
            A (int): represents the number of output actions that corresponds to the number of neurons in the output layer
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(S, 32),
            nn.ReLU(),
            nn.Linear(32, A)
        )
    
    def forward(self, x: Dict[str, np.ndarray]) -> th.Tensor:
        """
        Defines the forward pass of the model

        Args:
            x (Dict[str, np.ndarray]): a dictionary which contains the state to pass forward

        Returns:
            th.Tensor: model output
        """
        return self.model(x["state"])