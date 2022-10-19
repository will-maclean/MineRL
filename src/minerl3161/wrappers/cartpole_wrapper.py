import gym


class CartpoleWrapper(gym.ObservationWrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)
        self.observation_space = {"state": self.observation_space}
    
    def observation(self, observation):
        return {"state": observation}


def cartPoleWrapper(env, *args, **kwargs):
    return CartpoleWrapper(env, *args, **kwargs)