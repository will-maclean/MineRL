from typing import Dict, List, Optional

import numpy as np
import gym
import cv2


# TODO: write tests
class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_set: List[dict]) -> None:
        super().__init__(env)
        self.action_set = action_set

    def action(self, action):
        pass


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_name='pov') -> None:
        super().__init__(env)
        self.feature_name=feature_name
        self.rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        self.observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*self.observation_space[feature_name].shape[:-1], 1)
        )
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation[self.feature_name] = self._process(observation[self.feature_name])
        return observation
    
    def _process(self, rgb: np.ndarray):
        intermediate = rgb * self.rgb_weights
        return np.expand_dims(np.sum(intermediate, axis = 2), 2)


class Resize(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_name='pov', w=64, h=64) -> None:
        super().__init__(env)
        self.w = w
        self.h = h
        self.feature_name = feature_name

        self.observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.h, self.w, 3),
            dtype=self.observation_space.spaces[feature_name].dtype,
        )
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation[self.feature_name] = self._process(observation[self.feature_name])
        return observation
    
    def _process(self, frame: np.ndarray):
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return frame


class PyTorchImage(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, feature_name='pov') -> None:
        super().__init__(env)
        self.feature_name=feature_name
        self.observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.observation_space[feature_name].shape[2], self.observation_space[feature_name].shape[1], self.observation_space[feature_name].shape[0])
        )
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation[self.feature_name] = self._process(observation[self.feature_name])
        return observation
    
    def _process(self, feature: np.ndarray):
        return np.swapaxes(feature, 0, 2) / 255


class StackImage(gym.Wrapper):
    def __init__(self, env: gym.Env, feature_name='pov', frame=4) -> None:
        super().__init__(env)
        self.feature_name = feature_name
        self.frame = frame
        self.queue = np.zeros((self.frame, *self.env.observation_space[self.feature_name].shape[1:]))

        self.observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(frame, self.observation_space[feature_name].shape[1], self.observation_space[feature_name].shape[2])
        )
    
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
        self.queue[-1] = np.squeeze(next_state, axis=0)

        return self.queue


class InventoryFilter(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, features, feature_max=16) -> None:
        super().__init__(env)
        self.features = features
        self.feature_max = feature_max
        self.observation_space.spaces['inventory'] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(features),)
        )
    
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation['inventory'] = self._process(observation['inventory'])

        return observation
    
    def _process(self, inv_obs):
        inventory = []

        for key in self.features:
            inventory.append(
                min(inv_obs[key], self.feature_max) / self.feature_max
                )
        
        
        return np.array(inventory)


def mineRLObservationSpaceWrapper(
            env: gym.Env, 
            features: Optional[List[str]] = None,
            frame: int = 4, 
            downsize_width: int = 64, 
            downsize_height: int = 64
            ):
    if 'pov' in features:
        camera = True
        features.remove('pov')
    else:
        camera = False

    if features is not None:
        env = InventoryFilter(env, features=features)

    if camera:
        camera_feature_name = 'pov'
        env = Resize(env, feature_name=camera_feature_name, w=downsize_width, h=downsize_height)
        env = Grayscale(env, feature_name=camera_feature_name)
        env = PyTorchImage(env, feature_name=camera_feature_name)
        env = StackImage(env, frame=frame, feature_name=camera_feature_name)

    return env