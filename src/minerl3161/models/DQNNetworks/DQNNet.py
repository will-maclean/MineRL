from typing import Dict, List, Tuple, Union

import torch as th
from torch import nn
import numpy as np

from minerl3161.hyperparameters import MineRLDQNHyperparameters
from minerl3161.utils import sample_pt_state
from minerl3161.models.submodel import MineRLFeatureExtraction


class DQNNet(nn.Module):
    """
    Stores the PyTorch neural network to be used as a DQN network.
    """

    def __init__(
        self,
        state_shape: Dict[str, Tuple[int]],
        n_actions: int,
        dqn_hyperparams: MineRLDQNHyperparameters = None,
        layer_size: Union[int, None] = 64,
    ) -> None:
        """
        Initialiser for DQNNet

        Args:
            state_shape (Dict[str, Tuple[int]]): state shape to be used. Will only work with 1D states for now
            n_actions (int): number of actions available for the agent
            dqn_hyperparams (DQNHyperparameters):  the hyperparameters being used internally in this class
            layer_size (Union[int, None]): width of all Linear layers. Defaults to 64
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
        """
        Defines the forward pass of the model

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
    def _feature_names(state_shape: Dict[str, np.ndarray], dqn_hyperparams: MineRLDQNHyperparameters = None) -> List[str]:
        """
        Extracts feature names from either the state or the hyperparameters

        Args:
            state_shape (Dict[str, np.ndarray]): the state shape used by the feature extractor
            dqn_hyperparams (DQNHyperparameters): the hyperparameter object
        
        Returns:
            List[str]: features for the extractor to use
        """
        if dqn_hyperparams is not None:
            feature_names = dqn_hyperparams.feature_names
        else:
            try:
                feature_names = state_shape.spaces.keys()
            except AttributeError:
                feature_names = state_shape.keys()   
        
        return feature_names