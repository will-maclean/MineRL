from copy import deepcopy
import pickle
import random
from typing import Dict, Union

import numpy as np
import torch as th
import wandb

from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.models import RainbowDQN
from minerl3161.utils import epsilon_decay, np_dict_to_pt
from minerl3161.agents import BaseAgent


class RainbowDQNAgent(BaseAgent):
    """
    Rainbow Deep Q Learning algorithm that inherits from BaseAgent. 
    This includes a PyTorch neural network which acts as the function approximator.
    This algorithm implements the following improvements from the DQNAgent/DQNTrainer:
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
        RainbowDQNAgent initialiser

        Args:
            obs_space (Dict[str, np.ndarray]): environment observation space
            n_actions (int): number of actions in the action space
            device (str): PyTorch device to store agent on (generally either "cpu" for CPU training or "cuda:0" for GPU training)
            hyperparams (RainbowDQNHyperparameters): RainbowDQNHyperparameters instance stores specific hyperparameters for RainbowDQN training
            load_path (str): the path that a previously trained agent is stored which can be imported when training begins
        """
        super(RainbowDQNAgent, self).__init__()

        self.device = device
        self.hp = hyperparams
        self.n_actions = n_actions

        self.support = th.linspace(
            hyperparams.v_min, hyperparams.v_max, hyperparams.atom_size
        ).to(self.device)

        self.q1 = RainbowDQN(
            state_shape=obs_space,
            n_actions=n_actions, 
            dqn_hyperparams=hyperparams,
            support=self.support, 
            std_init=hyperparams.noisy_init
        ).to(self.device)
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)
    
    def watch_wandb(self) -> None:
        """
        Watch any relevant models with wandb
        """
        wandb.watch(self.q1)

    def act(self, state: Dict[str, np.ndarray], train: bool = False, step: int = None) -> Union[np.ndarray, dict]:
        """
        Chooses action from action space based on state

        Args:
            state (np.ndarray): environment state
            train (bool): determines if client code requires train or eval functionality (eval shoud not use eps)
            step (int): the current time step in training, used to determine current eps value

        Returns:
            np.ndarray, dict: chosen action, log dictionary
        """
        if train and self.hp.use_eps:
            eps = epsilon_decay(
                step,
                self.hp.eps_max,
                self.hp.eps_min,
                self.hp.eps_decay,
            )

            if random.random() < eps:
                action = th.randint(high=self.n_actions, size=(1,), device=self.device)
                return action, {"epsilon": eps}

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