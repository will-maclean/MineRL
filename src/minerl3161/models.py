from typing import Tuple
import torch as th
from torch import nn


# TODO: write tests
class DQNNet(nn.Module):
    def __init__(self, state_shape: Tuple[int], n_actions: int, layer_size=64) -> None:
        super().__init__()

        self.feature_extractor = lambda x: x  # TODO: create feature extractor based on state

        sample_input = th.ones((1, *state_shape))
        n_hidden_features = self.feature_extractor(sample_input).flatten(1).shape[1]

        # duelling architecture
        self.value = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size, bias=True),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size, bias=True),
            nn.ReLU(),
            nn.Linear(layer_size, n_actions)
        )

    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.feature_extractor(x)

        v = self.value(x)

        advantage = self.advantage(x)

        return v + (advantage - advantage.mean())