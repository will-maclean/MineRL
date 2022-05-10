import torch as th
from torch import nn


class DQNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        pass