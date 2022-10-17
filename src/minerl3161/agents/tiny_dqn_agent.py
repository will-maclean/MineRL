from copy import deepcopy
import pickle
import random
from typing import Union

import numpy as np
import torch as th
import wandb
from minerl3161.hyperparameters.dqn_hp import DQNHyperparameters

from minerl3161.models.models import TinyDQN
from minerl3161.utils.utils import epsilon_decay, np_dict_to_pt
from minerl3161.agents import BaseAgent


class TinyDQNAgent(BaseAgent):
    """
    Tiny version of the DQNAgent that inherits from the BaseAgent. This includes a PyTorch neural network.
    The neural network is a TinyDQN which is far smaller and simpler compared to the DQN network. This agent
    is more appropriate for simpler environments such as CartPole.
    """

    def __init__(
        self, 
        obs_space: int, 
        n_actions: int, 
        device: str, 
        hyperparams: DQNHyperparameters = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        TinyDQNAgent initialiser

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

        # Initialise policy network and create a deepcopy for the target network
        self.q1 = TinyDQN(S=obs_space["state"].shape[0], A=n_actions).to(device)
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)

    def act(self, state: np.ndarray, train: bool = False, step: int = None) -> Union[np.ndarray, dict]:
        """
        Chooses action from action space based on state

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

    def save(self, path: str) -> None:
        """
        Saves the current agent

        Args:
            path (str): path to save agent
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'TinyDQNAgent':
        """
        Loads agent

        Args:
            path (str): path to load from

        Returns:
            TinyDQNAgent: loaded TinyDQNAgent instance
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)
    
    def watch_wandb(self) -> None:
        """
        Watch any relevant models with wandb
        """
        wandb.watch(self.q1)
