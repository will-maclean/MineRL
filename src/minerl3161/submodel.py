import torch as pt
from torch import nn


class NothingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
