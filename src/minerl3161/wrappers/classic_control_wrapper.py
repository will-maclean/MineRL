from typing import Dict

import gym
import numpy as np


class ClassicControlWrapper(gym.ObservationWrapper):
    """
    Provides a simple wrapper for CartPole-like environments to be used with our code.
    """
    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env)
        self.observation_space = {"state": self.observation_space}
    
    def observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Modifies observation to be a dictionary

        Args:
            observation (np.ndarray): origional observation

        Returns:
            Dict[str, np.ndarray]: modified observation
        """
        return {"state": observation}


def classicControlWrapper(env: gym.Env, *args, **kwargs) -> ClassicControlWrapper:
    """
    Creates the cartpole wrapper

    Args:
        env (gym.Env): env to wrap

    Returns:
        CartpoleWrapper: wrapped env
    """
    return ClassicControlWrapper(env, *args, **kwargs)