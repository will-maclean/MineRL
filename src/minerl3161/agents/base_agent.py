from abc import ABC, abstractmethod
from typing import Union

import numpy as np


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
    
    @abstractmethod
    def watch_wandb(self):
        """watch any relevant models with wandb
        """
        pass

    @staticmethod
    def load(path: str):
        """Loads an agent from a path

        Args:
            path (str): path from which to load agent

        Returns:
            BaseAgent: loaded instance of an agent.
        """
        raise NotImplementedError()