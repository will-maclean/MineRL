from typing import Dict, Tuple, Union

import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.models.noisy_linear import NoisyLinear


class TinyRainbowDQN(nn.Module):
    """
    Stores the PyTorch neural network to be used as a TinyRainbowDQN network. This model contains NoisyLinear Layers
    intstead of Linear Layers as used in the Vanilla Policy.
    """

    def __init__(
        self, 
        state_shape: Dict[str, Tuple[int]],
        n_actions: int, 
        dqn_hyperparams: RainbowDQNHyperparameters,
        support: th.Tensor,
    ) -> None:
        super().__init__()
        
        self.support = support
        self.n_actions = n_actions
        self.atom_size = dqn_hyperparams.atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_shape["state"].shape[0], dqn_hyperparams.model_hidden_layer_size), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.noisy_init)
        self.advantage_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.n_actions * self.atom_size, dqn_hyperparams.noisy_init)

        # set value layer
        self.value_hidden_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.noisy_init)
        self.value_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.atom_size, dqn_hyperparams.noisy_init)
    
    def forward(self, x: Union[Dict[str, np.ndarray], np.ndarray]) -> th.Tensor:
        """
        Defines the forward pass of the model

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            x (Union[Dict[str, np.ndarray], np.ndarray]): state to pass forward, can either be passed in as a dictionary containing the state, or the
                                                          raw state itself

        Returns:
            th.Tensor: model output
        """
        if type(x) == dict:
            x = x["state"]
        
        dist = self.dist(x)
        q = th.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: th.Tensor) -> th.Tensor:
        """
        Determines the distributions for the atoms as per the C51 alogirthm (Distributional RL)

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            x (th.Tensor): state used to calculate distributions

        Returns:
            th.Tensor: model output
        """
        if type(x) == dict:
            x = x["state"]
        
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.n_actions, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """
        Used to reset the noisy layers

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need
        """
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()