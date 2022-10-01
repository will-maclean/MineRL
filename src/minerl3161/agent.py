"""Defines BaseAgent classes and all current implementations.
"""

from copy import deepcopy
import pickle
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models.models import DQNNet, TinyDQN
from minerl3161.utils import epsilon_decay, np_dict_to_pt

from minerl3161.pl_pretraining.pl_model import DQNPretrainer

from .hyperparameters import DQNHyperparameters


class BaseAgent(ABC):
    """Provides an abstract agent class for interacting with the minerl environments.

    Extending classes must implement the act() method, and must also implement saving
    and loading if that functionality will be required.
    """

    @abstractmethod
    def act(self, state: np.ndarray, train=False, step=None) -> Union[np.ndarray, dict]:
        """Chooses an action based on the given state

        Args:
            state (np.ndarray): the environment state
            train (bool): whether or not we are currently training
            step (Any): can store an environment step

        Returns:
            (np.ndarray, dict): the chosen action, info dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        """save the current agent

        Args:
            path (str): path in which to save the agent.
        """
        #TODO: should put the model on CPU before save
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
        obs_space: Dict[str, np.ndarray],
        n_actions: int,
        device: str,
        hyperparams: DQNHyperparameters,
        load_path: str = None,
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

        self.q1 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions,
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size,
        ).to(device)

        self.q2 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions,
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size,
        ).to(device)
        self.q2.load_state_dict(self.q1.state_dict())
        self.q2.requires_grad_(False)

        if load_path is not None:
            pl_model = DQNPretrainer.load_from_checkpoint(load_path, obs_space=obs_space, n_actions=n_actions, hyperparams=hyperparams)

            self.q1.load_state_dict(pl_model.q1.state_dict())
            self.q2.load_state_dict(pl_model.q2.state_dict())

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
