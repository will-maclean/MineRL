from copy import deepcopy
import pickle
from typing import Dict, Union

import torch as th
import numpy as np
import wandb

from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.models.DQNNetworks import TinyRainbowDQN
from minerl3161.utils import np_dict_to_pt
from minerl3161.agents import BaseAgent


class TinyRainbowDQNAgent(BaseAgent):
    """
    Tiny version of the RainbowDQNAgent that inherits from the BaseAgent. This includes a PyTorch neural network.
    The neural network is a TinyRainbowDQN which is far smaller and simpler compared to the RainbowDQN network. This agent
    is more appropriate for simpler environments such as CartPole. This algorithm implements the following improvements from the 
    DQNAgent/DQNTrainer:
        - Prioritised Experience Replay
        - Noisy Model Architecture
        - N-Step Learning
        - Distributional RL
    """

    def __init__(
        self,
        obs_space: Dict[str, np.ndarray],
        n_actions: int,
        device: str,
        hyperparams: RainbowDQNHyperparameters,
        load_path: str = None,
    ) -> None:
        """
        TinyRainbowDQNAgent initialiser

        Args:
            obs_space (Dict[str, np.ndarray]): environment observation space
            n_actions (int): number of actions in the action space
            device (str): PyTorch device to store agent on (generally either "cpu" for CPU training or "cuda:0" for GPU training)
            hyperparams (RainbowDQNHyperparameters): RainbowDQNHyperparameters instance stores specific hyperparameters for DQN training
            load_path (str): the path that a previously trained agent is stored which can be imported when training begins
        """

        super(TinyRainbowDQNAgent, self).__init__()

        self.device = device

        self.support = th.linspace(
            hyperparams.v_min, hyperparams.v_max, hyperparams.atom_size
        ).to(self.device)

        # Initialise policy network and create a deepcopy for the target network
        self.q1 = TinyRainbowDQN(
            state_shape=obs_space,
            n_actions=n_actions, 
            dqn_hyperparams=hyperparams,
            support=self.support, 
        ).to(self.device)

        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)

    def act(self, state: Dict[str, np.ndarray], train: bool = False, step: int = None) -> Union[np.ndarray, dict]:
        """
        Chooses action from action space based on state

        Args:
            state (np.ndarray): environment state

        Returns:
            np.ndarray, dict: chosen action, log dictionary
        """
        state = np_dict_to_pt(state, device=self.device, unsqueeze=True)

        selected_action = self.q1(state).argmax()  
        
        return selected_action, {}
    
    def save(self, path: str) -> None:
        """
        Saves the current agent

        Args:
            path (str): path to save agent
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'TinyRainbowDQNAgent':
        """
        Loads an agent from a path

        Args:
            path (str): path from which to load agent

        Returns:
            TinyRainbowDQNAgent: loaded instance of a TinyRainbowDQNAgent
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)    

    def watch_wandb(self) -> None:
        """
        Watch any relevant models with wandb
        """
        wandb.watch(self.q1)