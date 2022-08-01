from typing import Dict, List

import numpy as np
import gym


# TODO: write tests
class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_set: List[dict]) -> None:
        super().__init__(env)
        self.action_set = action_set

    def action(self, action):
        pass


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_name='camera') -> None:
        super().__init__(env)
        self.feature_name=feature_name
        self.rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation[self.feature_name] = self._process(observation)
        return super().observation(observation)
    
    def _process(self, rgb: np.ndarray):
        intermediate = np.dot(rgb, self.rgb_weights)
        return np.sum(intermediate, axis = 2)


class PyTorchImage(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_name='rgb') -> None:
        super().__init__(env)
        self.feature_name=feature_name
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation[self.feature_name] = self._process(observation)
        return super().observation(observation)
    
    def _process(self, feature: np.ndarray):
        return np.swapaxes(feature, 0, 2)


class StackImage(gym.Wrapper):
    def __init__(self, env: gym.Env, feature_name='rgb', frame=4) -> None:
        super().__init__(env)
        self.feature_name = feature_name
        self.frame = frame
        self.queue = np.zeros((self.frame, *self.env.observation_space[self.feature_name].shape))
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        next_state[self.feature_name] = self._obs(next_state[self.feature_name])

        return next_state, reward, done, info
    
    def reset(self):
        state = self.env.reset()

        state[self.feature_name] = self._obs(state[self.feature_name], reset=True)

        return state
    
    def _obs(self, next_state, reset=False):
        if reset:
            self.queue = np.zeros_like(self.queue)
        
        self.queue = np.roll(self.queue, shift=-1, axis=0)
        self.queue[-1] = next_state

        return self.queue


def mineRLObservationSpaceWrapper(env, frame=4, camera_feature_name='pov'):
    env = Grayscale(env, feature_name=camera_feature_name)
    env = PyTorchImage(env, feature_name=camera_feature_name)
    env = StackImage(env, frame=frame, feature_name=camera_feature_name)

    return env