"""Stores all PyTorch models
"""

from typing import Dict, Tuple

import torch as th
from torch import nn
import torch.nn.functional as F
from minerl3161.hyperparameters import DQNHyperparameters, RainbowDQNHyperparameters

from minerl3161.models.submodel import MineRLFeatureExtraction
from minerl3161.models.noisy_linear import NoisyLinear
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

        feature_names = DQNNet._feature_names(
            state_shape=state_shape,
            dqn_hyperparams=dqn_hyperparams
        )             

        self.feature_extractor = MineRLFeatureExtraction(
            observation_space=state_shape,
            feature_names=feature_names,
            mlp_hidden_size=dqn_hyperparams.mlp_output_size
            if dqn_hyperparams is not None
            else layer_size,
        )

        sample_input = sample_pt_state(
            observation_space=state_shape, 
            features=feature_names, 
            device="cpu",
            batch=1,
        )

        n_hidden_features = self.feature_extractor(sample_input).shape[1]

        # duelling architecture
        self.value = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(n_hidden_features, layer_size),
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
    
    @staticmethod
    def _feature_names(state_shape, dqn_hyperparams=None):
        if dqn_hyperparams is not None:
            feature_names = dqn_hyperparams.feature_names
        else:
            try:
                feature_names = state_shape.spaces.keys()
            except AttributeError:
                feature_names = state_shape.keys()   
        
        return feature_names


class TinyDQN(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(S, 32),
            nn.ReLU(),
            nn.Linear(32, A)
        )
    
    def forward(self, x):
        return self.model(x["state"])


class TinyRainbowDQN(nn.Module):
    def __init__(self, 
        state_shape: Dict[str, Tuple[int]],
        n_actions: int, 
        dqn_hyperparams: RainbowDQNHyperparameters,
        support: th.Tensor,
        std_init: float = 0.1
    ):
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
        self.advantage_hidden_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.model_hidden_layer_size, std_init)
        self.advantage_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.n_actions * self.atom_size, std_init)

        # set value layer
        self.value_hidden_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.model_hidden_layer_size, std_init)
        self.value_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.atom_size, std_init)
    
    def forward(self, x):
        if type(x) == dict:
            x = x["state"]
        
        dist = self.dist(x)
        q = th.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: th.Tensor) -> th.Tensor:
        """Get distribution for atoms."""
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
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()



class RainbowDQN(nn.Module):
    def __init__(self, 
        state_shape: Dict[str, Tuple[int]],
        n_actions: int, 
        dqn_hyperparams: RainbowDQNHyperparameters,
        support: th.Tensor,
        std_init: float = 0.1
    ):
        super().__init__()
        
        self.support = support
        self.n_actions = n_actions
        self.atom_size = dqn_hyperparams.atom_size

        feature_names = DQNNet._feature_names(
            state_shape=state_shape,
            dqn_hyperparams=dqn_hyperparams
        )             

        self.feature_extractor = MineRLFeatureExtraction(
            observation_space=state_shape,
            feature_names=feature_names,
            mlp_hidden_size=dqn_hyperparams.mlp_output_size
            if dqn_hyperparams is not None
            else 64,
        )

        sample_input = sample_pt_state(
            observation_space=state_shape, 
            features=feature_names, 
            device="cpu",
            batch=1,
        )

        n_hidden_features = self.feature_extractor(sample_input).shape[1]
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(n_hidden_features, dqn_hyperparams.model_hidden_layer_size, std_init)
        self.advantage_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.n_actions * self.atom_size, std_init)

        # set value layer
        self.value_hidden_layer = NoisyLinear(n_hidden_features, dqn_hyperparams.model_hidden_layer_size, std_init)
        self.value_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.atom_size, std_init)
    
    def forward(self, x):
        dist = self.dist(x)
        q = th.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: th.Tensor) -> th.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_extractor(x)
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
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

