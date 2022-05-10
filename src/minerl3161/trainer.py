import numpy as np
import torch
from torch.optim import Optimizer, Adam
import gym

from agent import BaseAgent
from hyperparameters import BaseHyperparameters
from buffer import ReplayBuffer
from evaluator import Evaluator

class BaseTrainer:
    def __init__(self, env: gym.Env, agent: BaseAgent, hyperparameters: BaseHyperparameters) -> None:
        self.env = env
        self.agent = agent
        self.hp = hyperparameters
        self.gathered_transitions = ReplayBuffer(self.hp.buffer_size_gathered)
        self.dataset_transitions = ReplayBuffer(self.hp.buffer_size_dataset)
        self.evaluator = Evaluator(env)
    
    def train() -> None:
        pass

    def _gather(steps: int) -> None:
        pass

    def _train_step(step: int) -> None:
        pass

    def _housekeeping(step: int) -> None:
        # TODO: Better name for this method
        pass


class DQNTrainer(BaseTrainer):
    def __init__(self, optim: Optimizer = None) -> None:
        self.optim = optim if optim is not None else Adam(self.agent.q1.parameters(), lr=self.hp.lr)
    
    def _calc_loss(batch: dict) -> torch.Tensor:
        pass
