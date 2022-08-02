from typing import List, Dict
import numpy as np
import gym


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, filepath: str = "src/actions/actions.npy") -> None:
        super().__init__(env)
        self.action_set = np.load(filepath)

    def get_action(self, action_idx) -> Dict[str, List[float]]:
        return {
                "vector": self.action_set[action_idx]
            }
    
    def get_actions_count(self) -> int:
        return len(self.action_set)