from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import gym
import cv2
import os
import pickle

import minerl3161


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, functional_acts: bool = True, extracted_acts: bool = True) -> None:
        super().__init__(env)
        extracted_acts_filename = "extracted-actions.pickle"
        functional_acts_filename = "functional-actions.pickle"
        self.action_set = []

        if extracted_acts:
            e_filepath = os.path.join(minerl3161.actions_path, extracted_acts_filename)
            with open(e_filepath, "rb") as f:
                self.action_set.extend(pickle.load(f))

        if functional_acts:
            f_filepath = os.path.join(minerl3161.actions_path, functional_acts_filename)
            with open(f_filepath, "rb") as f:
                self.action_set.extend(pickle.load(f))
        
        self.action_space = gym.spaces.Discrete(len(self.action_set))

    def action(self, action_idx: int) -> Dict[str, str]:
        return self.action_set[action_idx]
    
    def get_actions_count(self) -> int:
        return len(self.action_set)


def convert_action(action=None, action_space=None):
    return action, action_space


def obs_grayscale(state=None, observation_space=None, feature_name='pov'):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(observation_space[feature_name].shape[:-1], 1)
        )
    
    if state is not None:
        rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        intermediate = state[feature_name] * rgb_weights
        state[feature_name] = np.expand_dims(np.sum(intermediate, axis = 2), 2)

    return state, observation_space


def obs_resize(state=None, observation_space=None, feature_name='pov', w=64, h=64):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=observation_space[feature_name].dtype,
        )
    
    if state is not None:
        state[feature_name] = cv2.resize(state[feature_name], (w, h), interpolation=cv2.INTER_AREA)
    
    return state, observation_space


def obs_pytorch_image(state=None, observation_space=None, feature_name='pov'):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(observation_space[feature_name].shape[2], observation_space[feature_name].shape[1], observation_space[feature_name].shape[0])
        )

    if state is not None:
        state = np.swapaxes(state[feature_name], 0, 2) / 255, observation_space
    
    return state, observation_space


def obs_stack_image(state=None, observation_space=None, feature_name='pov', state_buffer=None, frame=4):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(frame, observation_space[feature_name].shape[1], observation_space[feature_name].shape[2])
        )

    if state is not None:
        if state_buffer is None:
            state_buffer = np.zeros((frame, *observation_space[feature_name].shape[1:]))

        state_buffer = np.roll(state_buffer, shift=-1, axis=0)
        state_buffer[-1] = np.squeeze(state[feature_name], axis=0)

    return state_buffer, observation_space, state_buffer


def obs_inventory_filter(state=None, observation_space=None, features=None, feature_max=16):
    if observation_space is not None:
        if len(features) == 1 and features[0] == "all":
            # return all the items
            observation_space.spaces['inventory'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(len(list(observation_space['inventory'].spaces.keys())),)
            )
        elif len(features) > 0 and "all" not in features:
            # return a specified set of features
            observation_space.spaces['inventory'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(len(features),)
            )
        else:
            raise ValueError(f"features must be either ['all'] or a list of features not containing 'all'. Features is: {features}")

    if state is not None:
        inventory = []

        if len(features) == 1 and features[0] == "all":
            # return all the items
            for key in state['inventory']:
                inventory.append(
                    min(state['inventory'][key], feature_max) / feature_max
                    )
            
        elif len(features) > 0 and "all" not in features:
            # return a specified set of features
            for key in features:
                inventory.append(
                    min(state['inventory'][key], feature_max) / feature_max
                    )
            
        else:
            raise ValueError(f"features must be either ['all'] or a list of features not containing 'all'. Features is: {features}")
        
        state['inventory'] = np.array(inventory)

    return state, observation_space


def obs_toggle_equipped_items(state=None, observation_space=None, include_equipped_items=False):
    if observation_space is not None:
        if not include_equipped_items:
            del observation_space.spaces["equipped_items"]

    if state is not None:
        if not include_equipped_items:
            del state["equipped_items"]

    return state, observation_space


def convert_state(state=None, observation_space=None, *args, **kwargs):
    state, observation_space = obs_inventory_filter(state=state, observation_space=observation_space, *args, **kwargs)
    state, observation_space = obs_toggle_equipped_items(state=state, observation_space=observation_space, *args, **kwargs)
    state, observation_space = obs_resize(state=state, observation_space=observation_space, *args, **kwargs)
    state, observation_space = obs_grayscale(state=state, observation_space=observation_space, *args, **kwargs)
    state, observation_space, state_buffer = obs_pytorch_image(state=state, observation_space=observation_space, *args, **kwargs)
    state, observation_space = obs_stack_image(state=state, observation_space=observation_space, *args, **kwargs)

    return state, observation_space, state_buffer


class MineRLWrapper(gym.Wrapper):
    def __init__(self, env, features=None, include_equipped_items=False, resize_w=64, resize_h=64, img_feature_name="pov", n_stack=4) -> None:
        super().__init__(env)

        # update action space
        _, self.action_space = convert_action(action_space=self.action_space)

        # update observation space
        _, self.observation_space, _ = convert_state(observation_space=self.observation_space, features=features, include_equipped_items=include_equipped_items, resize_w=resize_w, resize_h=resize_h, img_feature_name=img_feature_name, n_stack=n_stack)

        # create state buffer
        self.state_buffer = np.zeros((n_stack, resize_h, resize_w), dtype=np.float32)
    
    def reset(self):
        state = self.env.reset()
        state, _, self.state_buffer = convert_state(state=state, state_buffer=self.state_buffer)
        return state
    
    def step(self, action):
        action, _ = convert_action(action=action)
        state, reward, done, info = self.env.step(action)
        state, _, self.state_buffer = convert_state(state=state, state_buffer=self.state_buffer)

        return state, reward, done, info


def minerlWrapper(env, *args, **kwargs):
    return MineRLWrapper(env, *args, **kwargs)
