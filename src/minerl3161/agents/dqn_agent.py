import pickle
from typing import Dict, Union
import random

import numpy as np
import wandb
import torch as th

from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models import DQNNet
from minerl3161.pl_pretraining.pl_model import DQNPretrainer
from minerl3161.utils import epsilon_decay, np_dict_to_pt
from minerl3161.agents import BaseAgent


class DQNAgent(BaseAgent):
    """
    Deep Q Learning algorithm that inherits from BaseAgent. 
    This includes a PyTorch neural network which acts as the function approximator.
    """

    def __init__(
        self,
        obs_space: Dict[str, np.ndarray],
        n_actions: int,
        device: str,
        hyperparams: DQNHyperparameters,
        load_path: str = None,
    ) -> None:
        """
        DQNAgent initialiser

        Args:
            obs_space (Dict[str, np.ndarray]): environment observation space
            n_actions (int): number of actions in the action space
            device (str): PyTorch device to store agent on (generally either "cpu" for CPU training or "cuda:0" for GPU training)
            hyperparams (DQNHyperparameters): DQNHyperparameters instance stores specific hyperparameters for DQN training
            load_path (str): the path that a previously trained agent is stored which can be imported when training begins
        """
        super().__init__()
        self.device = device
        self.hp = hyperparams

        self.obs_space = obs_space
        self.n_action = n_actions

        # Initialise policy network
        self.q1 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions,
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size,
        ).to(device)

        # Initialise target network
        self.q2 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions,
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size,
        ).to(device)
        self.q2.load_state_dict(self.q1.state_dict())
        self.q2.requires_grad_(False)

        # If a load path has been specified, load the weights into the initialised models
        if load_path is not None:
            pl_model = DQNPretrainer.load_from_checkpoint(load_path, obs_space=obs_space, n_actions=n_actions, hyperparams=hyperparams)

            self.q1.load_state_dict(pl_model.q1.state_dict())
            self.q2.load_state_dict(pl_model.q2.state_dict())
    
    def watch_wandb(self) -> None:
        """
        Watch any relevant models with wandb
        """
        wandb.watch(self.q1)
        wandb.watch(self.q2)

    def act(self, state: np.ndarray, train: bool = False, step: int = None) -> Union[np.ndarray, dict]:
        """
        Chooses action from action space based on state

        Args:
            state (np.ndarray): environment state
            train (bool): determines if client code requires train or eval functionality (eval shoud not use eps)
            step (int): the current time step in training, used to determine current eps value

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
    def save(self, path: str) -> None:
        """
        Saves the current agent

        Args:
            path (str): path to save agent
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'DQNAgent':
        """
        Loads an agent from a path

        Args:
            path (str): path from which to load agent

        Returns:
            DQNAgent: loaded instance of a DQNAgent
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)