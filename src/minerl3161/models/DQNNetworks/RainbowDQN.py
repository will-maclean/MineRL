from typing import Dict, Tuple

import torch as th
from torch import nn
import torch.nn.functional as F

from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.utils import sample_pt_state
from minerl3161.models.submodel import MineRLFeatureExtraction
from minerl3161.models.noisy_linear import NoisyLinear
from minerl3161.models.DQNNetworks.DQNNet import DQNNet


class RainbowDQN(nn.Module):
    """
    Stores the PyTorch neural network to be used as a RainbowDQN network. This model contains NoisyLinear Layers
    intstead of Linear Layers as used in the Vanilla Policy.
    """

    def __init__(
        self, 
        state_shape: Dict[str, Tuple[int]],
        n_actions: int, 
        dqn_hyperparams: RainbowDQNHyperparameters,
        support: th.Tensor,
    ) -> None:
        """
        Initialiser for RainbowDQN

        Args:
            state_shape (Dict[str, Tuple[int]]): state shape to be used. Will only work with 1D states for now
            n_actions (int): number of actions available for the agent
            dqn_hyperparams (RainbowDQNHyperparameters):  the hyperparameters being used internally in this class
            support (th.Tensor): the minimum and maximum range of the reward scale in the C51 algorithm
        """
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
        self.advantage_hidden_layer = NoisyLinear(n_hidden_features, dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.noisy_init)
        self.advantage_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.n_actions * self.atom_size, dqn_hyperparams.noisy_init)

        # set value layer
        self.value_hidden_layer = NoisyLinear(n_hidden_features, dqn_hyperparams.model_hidden_layer_size, dqn_hyperparams.noisy_init)
        self.value_layer = NoisyLinear(dqn_hyperparams.model_hidden_layer_size, self.atom_size, dqn_hyperparams.noisy_init)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Defines the forward pass of the model

        TODO: licence

        Args:
            x (th.Tensor): state to pass forward

        Returns:
            th.Tensor: model output
        """
        dist = self.dist(x)
        q = th.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: th.Tensor) -> th.Tensor:
        """
        Determines the distributions for the atoms as per the C51 alogirthm (Distributional RL)

        TODO: licence

        Args:
            x (th.Tensor): state used to calculate distributions

        Returns:
            th.Tensor: model output
        """
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
    
    def reset_noise(self) -> None:
        """
        Used to reset the noisy layers

        TODO: licence
        """
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()