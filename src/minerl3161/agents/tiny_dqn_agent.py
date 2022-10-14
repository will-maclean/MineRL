from copy import deepcopy
import pickle
import random
from typing import Union

import numpy as np
import torch as th
import wandb

from minerl3161.models.models import TinyDQN
from minerl3161.utils.utils import epsilon_decay, np_dict_to_pt
from minerl3161.agents import BaseAgent


class TinyDQNAgent(BaseAgent):
    """BaseAgent implementation that implements a Deep Q Learning algorithm. This include a PyTorch neural network."""

    def __init__(
        self, obs_space: int, n_actions: int, device: str, hyperparams=None, *args, **kwargs
    ) -> None:
        """Base agent initialiser

        Args:
            obs_space (Dict[str, np.ndarray]): environment observation space
            n_actions (int): number of actions in the action space
            device (str): PyTorch device to store agent on (generally either "cpu" for CPU training or "cuda:0" for GPU training)
            hyperparams (DQNHyperparameters): DQNHyperparameters instance stores specific hyperparameters for DQN training
        """
        super().__init__()
        self.device = device

        self.hp = hyperparams

        self.obs_space = obs_space
        self.n_action = n_actions

        self.q1 = TinyDQN(S=obs_space["state"].shape[0], A=n_actions).to(device)
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)

    def act(self, state: np.ndarray, train=False, step=None) -> Union[np.ndarray, dict]:
        """chooses action from action space based on state

        Args:
            state (np.ndarray): environment state

        Returns:
            np.ndarray, dict: chosen action, log dictionary
        """
        state = np_dict_to_pt(state, device=self.device, unsqueeze=True)

        if train:
            eps = epsilon_decay(
                step,
                self.hp.eps_max,
                self.hp.eps_min,
                self.hp.eps_decay,
            )

            if random.random() < eps:
                action = th.randint(high=self.n_action, size=(1,), device=self.device)
            else:
                action = self.q1(state).argmax()

            return action, {"epsilon": eps}
        else:
            with th.no_grad():
                q_vals = self.q1(state)

                action = q_vals.squeeze().argmax().detach().cpu().numpy()

                return action, {}

    def save(self, path: str):
        """saves the current agent

        Args:
            path (str): path to save agent
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        """Load agent

        Args:
            path (str): path to load from

        Returns:
            DQNAgent: loaded DQNAgent instance
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)
    
    def watch_wandb(self):
        wandb.watch(self.q1)
