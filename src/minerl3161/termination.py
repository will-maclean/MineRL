from abc import abstractmethod

import numpy as np


class TerminationCondition:
    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        pass


class AvgEpisodeReturnTerminationCondition(TerminationCondition):
    def __init__(self, termination_avg=100, window=10) -> None:
        super().__init__()

        assert termination_avg > 0, "Only termination averages > 0 are currently supported"

        self.termination_avg = termination_avg
        self.window = window

        self.buffer = np.zeros(self.window)
    
    def __call__(self, episode_return, *args, **kwargs):
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


def get_termination_condition(env_name):
    try:
        init_dict_list =  environment_termination_conditions[env_name]

        conditions = []

        for condition in init_dict_list:

            c = condition["class"](**condition["kwargs"])

            conditions.append(c)
        
        return conditions

    except KeyError:
        return [] 
