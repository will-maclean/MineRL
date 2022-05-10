from abc import ABC, abstractmethod
from copy import deepcopy
import random
from typing import Tuple, Union

import numpy as np
import torch as th

from minerl3161.models import DQNNet


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path):
        raise NotImplementedError()
    
    @abstractmethod
    @staticmethod
    def load(self, state):
        raise NotImplementedError()


class DQNAgent(BaseAgent):
    def __init__(self, state_shape: Tuple[int], n_actions: int, device: str, hyperparams: Hyperparameters) -> None:  # FIXME @ Jade
        super().__init__()
        self.device = device
        self.hyperparams = hyperparams

        self.q1 = DQNNet(state_shape, n_actions, hyperparams.model_hidden_layer_size).to(device)  # FIXME @ Jade
        
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)
        self.q2.eval()
    
    def act(self, state: np.ndarray) -> np.ndarray:
        state = th.from_numpy(state).to(self.device).unsqueeze(0)

        q_vals = self.q1(state)

        action = q_vals.squeeze().argmax().detach().cpu().numpy()

        return action
    
    def eps_greedy_act(self, state: Union[np.ndarray, th.Tensor], step: int) -> Union[np.ndarray, th.Tensor]:
        if type(state) == np.np.ndarray:
            state = th.from_numpy(state).to(self.device).unsqueeze(0)
            was_np = True
        else:
            was_np = False
        

        eps = epsilon_decay()  # TODO

        if random.random() < eps:
            pass
        else:
            pass

        if was_np:
            action = action.detach().cpu().numpy()
        
        return action
    
    
    def save(self, path):
        raise NotImplementedError()
    
    @staticmethod
    def load(self, state):
        raise NotImplementedError()
