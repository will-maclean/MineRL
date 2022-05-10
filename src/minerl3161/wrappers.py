from typing import List
import gym

class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_set: List[dict]) -> None:
        super().__init__(env)
        self.action_set = action_set
    
    def action(self, action):
        pass