from copy import deepcopy
import pickle
from typing import Dict, Union

import torch as th
import numpy as np
import wandb

from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.models import TinyRainbowDQN
from minerl3161.utils import np_dict_to_pt
from minerl3161.agents import BaseAgent


class TinyRainbowDQNAgent(BaseAgent):
    def __init__(
        self,
        obs_space: Dict[str, np.ndarray],
        n_actions: int,
        device: str,
        hyperparams: RainbowDQNHyperparameters,
        load_path: str = None,
    ):
        super(TinyRainbowDQNAgent, self).__init__()

        self.device = device

        self.support = th.linspace(
            hyperparams.v_min, hyperparams.v_max, hyperparams.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.q1 = TinyRainbowDQN(
            state_shape=obs_space,
            n_actions=n_actions, 
            dqn_hyperparams=hyperparams,
            support=self.support, 
            std_init=hyperparams.noisy_init
        ).to(self.device)
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)

    def act(self, state: Dict[str, np.ndarray], train=False, step=None) -> Union[np.ndarray, dict]:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection required

        state = np_dict_to_pt(state, device=self.device, unsqueeze=True)

        selected_action = self.q1(state).argmax()  
        
        return selected_action, {}
    
    def save(self, path: str):
        """saves the current agent

        Args:
            path (str): path to save agent
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    def watch_wandb(self):
        wandb.watch(self.q1)