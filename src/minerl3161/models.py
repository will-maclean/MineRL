"""Stores all PyTorch models
"""

from typing import Dict, Tuple

import torch as th
from torch import nn
from minerl3161.hyperparameters import DQNHyperparameters

from minerl3161.submodel import MineRLFeatureExtraction
from minerl3161.utils import sample_pt_state


# TODO: write tests
class DQNNet(nn.Module):
    """stores the PyTorch neural network to be used as a DQN network."""

    def __init__(
        self,
        state_shape: Dict[str, Tuple[int]],
        n_actions: int,
        dqn_hyperparams: DQNHyperparameters = None,
        layer_size=64,
    ) -> None:
        """intialiser for DQNNet

        Args:
            state_shape (Tuple[int]): state shape to be used. Will only work with 1D states for now
            n_actions (int): number of actions available for the agent
            layer_size (int, optional): width of all Linear layers. Defaults to 64.
        """
        super().__init__()

        self.feature_extractor = MineRLFeatureExtraction(
            state_shape,
            feature_names=dqn_hyperparams.feature_names
            if dqn_hyperparams is not None
            else None,
            mlp_hidden_size=dqn_hyperparams.mlp_output_size
            if dqn_hyperparams is not None
            else None,
        )

        sample_input = sample_pt_state(
            state_shape, 
            dqn_hyperparams.feature_names
            if dqn_hyperparams is not None
            else state_shape.spaces.keys(), 
            device="cpu",
            batch=1,
        )

        n_hidden_features = self.feature_extractor(sample_input).shape[1]

        # duelling architecture
        self.value = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size, bias=True),
            nn.ReLU(),
            nn.Linear(layer_size, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size, bias=True),
            nn.ReLU(),
            nn.Linear(layer_size, n_actions),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """forward pass of the model

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        x = self.feature_extractor(x)

        v = self.value(x)

        advantage = self.advantage(x)

        return v + (advantage - advantage.mean())
