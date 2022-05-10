from abc import ABC, abstractmethod
import numpy as np

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
    def __init__(self) -> None:
        super().__init__()
        self.model = DQNNet()
    
    def act(self, state):
        raise NotImplementedError()
    
    def save(self, path):
        raise NotImplementedError()
    
    @staticmethod
    def load(self, state):
        raise NotImplementedError()
