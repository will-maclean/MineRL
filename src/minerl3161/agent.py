"""Defines BaseAgent classes and all current implementations.
"""

import pickle
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, Union, Dict

import numpy as np
import torch as th
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models import DQNNet
from minerl3161.utils import epsilon_decay

from .hyperparameters import DQNHyperparameters


class BaseAgent(ABC):
    """Provides an abstract agent class for interacting with the minerl environments.

    Extending classes must implement the act() method, and must also implement saving
    and loading if that functionality will be required.
    """

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Chooses an action based on the given state

        Args:
            state (np.ndarray): the environment state

        Returns:
            action (np.ndarray): the chosen action
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        """save the current agent

        Args:
            path (str): path in which to save the agent.
        """
        raise NotImplementedError()

    @staticmethod
    def load(path: str):
        """Loads an agent from a path

        Args:
            path (str): path from which to load agent

        Returns:
            BaseAgent: loaded instance of an agent.
        """
        raise NotImplementedError()


# TODO: write tests
class DQNAgent(BaseAgent):
    """BaseAgent implementation that implements a Deep Q Learning algorithm. This include a PyTorch neural network."""

    def __init__(
        self,
        state_space: Dict[str, np.ndarray],
        n_actions: int,
        device: str,
        hyperparams: DQNHyperparameters,
    ) -> None:
        """Base agent initialiser

        Args:
            state_space (Tuple[int]): shape of the state shape dimensions
            n_actions (int): number of actions in the action space
            device (str): PyTorch device to store agent on (generally either "cpu" for CPU training or "cuda:0" for GPU training)
            hyperparams (DQNHyperparameters): DQNHyperparameters instance stores specific hyperparameters for DQN training
        """
        super().__init__()
        self.device = device
        self.hyperparams = hyperparams

        self.state_shape = state_space
        self.n_action = n_actions

        self.q1 = DQNNet(
            state_space, n_actions, hyperparams.model_hidden_layer_size
        ).to(device)

        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)
        self.q2.eval()

    def act(self, state: np.ndarray) -> np.ndarray:
        """chooses action from action space based on state

        Args:
            state (np.ndarray): environment state

        Returns:
            np.ndarray: chosen action
        """
        state = th.from_numpy(state).to(self.device).unsqueeze(0)

        q_vals = self.q1(state)

        action = q_vals.squeeze().argmax().detach().cpu().numpy()

        return action

    def eps_greedy_act(
        self, state: Union[np.ndarray, th.Tensor], step: int
    ) -> Union[np.ndarray, th.Tensor]:
        """Chooses an action under the epsilon greedy policy

        Args:
            state (Union[np.ndarray, th.Tensor]): environment
            step (int): training step

        Returns:
            Union[np.ndarray, th.Tensor]: chosen action
        """
        if type(state) == np.ndarray:
            state = th.from_numpy(state).to(self.device).unsqueeze(0)
            was_np = True
        else:
            was_np = False

        eps = epsilon_decay(
            step,
            self.hyperparams.eps_max,
            self.hyperparams.eps_min,
            self.hyperparams.eps_decay,
        )

        if random.random() < eps:
            action = th.randint(high=self.n_action, size=(1,), device=self.device)
        else:
            action = self.q1.argmax()

        if was_np:
            action = action.detach().cpu().numpy()

        return action

    # TODO: Determine if pickle supports saving and loading of model weights
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
