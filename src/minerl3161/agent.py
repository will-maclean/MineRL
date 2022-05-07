from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state):
        raise NotImplementedError()


class DQNAgent(BaseAgent):
    def act(self, state):
        pass
