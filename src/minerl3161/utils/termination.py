from abc import abstractmethod
from typing import List

import numpy as np


class TerminationCondition:
    """Provides an interface to terminate training
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        """Call and determine whether to terminate training

        Returns:
            bool: true if training should terminate, false otherwise
        """
        pass


class AvgEpisodeReturnTerminationCondition(TerminationCondition):
    """Determines whether trainign should terminate, based on average episode return
    """

    def __init__(self, termination_avg: float = 100, window: int = 10) -> None:
        """_summary_

        Args:
            termination_avg (float, optional): average return that must be reached. Defaults to 100.
            window (int, optional): window size for running average. Defaults to 10.
        """
        super().__init__()

        assert termination_avg > 0, "Only termination averages > 0 are currently supported"

        self.termination_avg = termination_avg
        self.window = window

        self.buffer = np.zeros(self.window)
    
    def __call__(self, episode_return, *args, **kwargs):
        """determines whether episode return is high enough

        Args:
            episode_return (float): return from an episode

        Returns:
            bool: true if training should terminate, false otherwise
        """
        self.buffer = np.roll(self.buffer, shift=-1)
        self.buffer[-1] = episode_return

        return self.buffer.mean() >= self.termination_avg


environment_termination_conditions = {
    "CartPole-v0": [
        {
            "class": AvgEpisodeReturnTerminationCondition,
            "kwargs": {
                "termination_avg": 180,
                "window": 10
            }
        }
    ],
    "CartPole-v1": [
        {
            "class": AvgEpisodeReturnTerminationCondition,
            "kwargs": {
                "termination_avg": 450,
                "window": 10
            }
        }
    ],
}


def get_termination_condition(env_name: str) -> List[TerminationCondition]:
    """Gets a list of termination conditions based on the name of the environment

    Args:
        env_name (str): name of the environment

    Returns:
        List[TerminationCondition]: list of necessary termination conditions.
    """
    try:
        init_dict_list =  environment_termination_conditions[env_name]

        conditions = []

        for condition in init_dict_list:

            c = condition["class"](**condition["kwargs"])

            conditions.append(c)
        
        return conditions

    except KeyError:
        return [] 
