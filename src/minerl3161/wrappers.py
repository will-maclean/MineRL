from copy import deepcopy
import stat
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import gym
import cv2
import os
import pickle

import minerl3161

def decode_action(obj: dict, camera_shrink_factor=100) -> list:
    """Decodes an action to fit into a dataframe.
    Helper function for MineRLWrapper.map_action()
    """
    proc = {
        'attack': 0,
        'back': 0,
        'camera0': 0.0,
        'camera1': 0.0,
        'craft': 'none',
        'equip': 'none',
        'forward': 0,
        'jump': 0,
        'left': 0,
        'nearbyCraft': 'none',
        'nearbySmelt': 'none',
        'place': 'none',
        'right': 0,
        'sneak': 0,
        'sprint': 0
    }

    for k in obj.keys():
        if k == "camera":
            for d, dim in enumerate(obj[k]):
                proc[f"{k}{d}"] = dim/camera_shrink_factor
        else:
            proc[k] = obj[k] if not isinstance(obj[k], np.ndarray) else obj[k].tolist()
    return proc

def obs_grayscale(state=None, observation_space=None, feature_name='pov', *args, **kwargs):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*observation_space[feature_name].shape[:-1], 1)
        )
    
    if state is not None:
        rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        intermediate = state[feature_name] * rgb_weights
        state[feature_name] = np.expand_dims(np.sum(intermediate, axis = 2), 2)

    return state, observation_space


def obs_resize(state=None, observation_space=None, feature_name='pov', resize_w=64, resize_h=64, *args, **kwargs):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(resize_w, resize_h, 3),
            dtype=observation_space[feature_name].dtype,
        )
    
    if state is not None:
        state[feature_name] = cv2.resize(state[feature_name], (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    
    return state, observation_space


def obs_pytorch_image(state=None, observation_space=None, feature_name='pov', *args, **kwargs):
    if observation_space is not None:
        observation_space.spaces[feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(observation_space[feature_name].shape[2], observation_space[feature_name].shape[1], observation_space[feature_name].shape[0])
        )

    if state is not None:
        state[feature_name] = np.swapaxes(state[feature_name], 0, 2) / 255
    
    return state, observation_space


def obs_stack_image(state: Dict[str, np.ndarray] = None, observation_space=None, feature_name='pov', state_buffer=None, frame=4, *args, **kwargs):
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
        new_state = state[feature_name]
        state_buffer[-1] = np.squeeze(new_state, axis=0)

        state[feature_name] = state_buffer.copy()

    return state, observation_space, state_buffer


def obs_inventory_filter(state=None, observation_space=None, features=None, feature_max=16, *args, **kwargs):
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


def obs_toggle_equipped_items(state=None, observation_space=None, include_equipped_items=False, *args, **kwargs):
    if observation_space is not None:
        if not include_equipped_items:
            # we need to make a copy of the observation space and use that instead so that we don't modify the original
            # observation space, as deleting is in-place
            observation_space = deepcopy(observation_space)
            del observation_space.spaces["equipped_items"]

    if state is not None:
        if not include_equipped_items:
            del state["equipped_items"]

    return state, observation_space


class MineRLWrapper(gym.Wrapper):
    def __init__(self, 
                env, 
                features=None, 
                include_equipped_items=False, 
                resize_w=64, 
                resize_h=64, 
                img_feature_name="pov", 
                n_stack=4,
                functional_acts=True,
                extracted_acts=True,
        ) -> None:
        super().__init__(env)

        self.obs_kwargs = {
            "features": features if features is not None else ["all"],
            "resize_w": resize_w,
            "resize_h": resize_h,
            "img_feature_name": img_feature_name,
            "n_stack": n_stack,
            "include_equipped_items": include_equipped_items,
            "state_buffer": np.zeros((n_stack, resize_h, resize_w), dtype=np.float32),
            "last_unprocessed_state": None,
        }

        # update action space
        self.action_set = MineRLWrapper.create_action_set(functional_acts=functional_acts, extracted_acts=extracted_acts)
        _, self.action_space = MineRLWrapper.convert_action(action_space=self.action_space, action_set=self.action_set)

        # update observation space
        _, self.observation_space, _ = MineRLWrapper.convert_state(observation_space=deepcopy(self.observation_space), **self.obs_kwargs)
    
    def reset(self):
        self.obs_kwargs["state_buffer"] = np.zeros_like(self.obs_kwargs["state_buffer"])
        
        self.obs_kwargs["last_unprocessed_state"] = self.env.reset()
        state, _, self.obs_kwargs["state_buffer"] = MineRLWrapper.convert_state(state=self.obs_kwargs["last_unprocessed_state"], **self.obs_kwargs)
        return state
    
    def step(self, action):
        action, _ = MineRLWrapper.convert_action(action=action, last_unprocessed_state=self.obs_kwargs["last_unprocessed_state"], action_set=self.action_set)
        self.obs_kwargs["last_unprocessed_state"], reward, done, info = self.env.step(action)
        state, _, self.obs_kwargs["state_buffer"] = MineRLWrapper.convert_state(state=self.obs_kwargs["last_unprocessed_state"], **self.obs_kwargs)

        return state, reward, done, info

    @staticmethod
    def map_action(obs:dict, action_set: list) -> int:
        """ Maps an observation from the env/dataset to an action index in our action set
        Args:
            obs (dict): A single action from the env/dataset in dictionary form
            action_set (dict): The action set initialised by the MineRLWrapper
        """
        cluster_centers = pd.DataFrame([decode_action(i) for i in action_set])

        # First checks if the action is categorical in nature
        cat_list = ['place', 'nearbyCraft', 'nearbySmelt', 'craft', 'equip']
        for cat_act in cat_list:
            if obs[cat_act] != 'none':
                return cluster_centers[cluster_centers[cat_act] == obs[cat_act]].index[0]

        # The values of each numerical field in a list
        obs_num = list({k: v for k, v in obs.items() if k not in cat_list}.values())

        # Calculates the euclidean distance between `obs` and every action in action set
        distances = [
            np.linalg.norm(obs_num - action.values) for _, action in cluster_centers.drop(
                cat_list, axis=1).iterrows()
        ]
        return np.argmin(distances)

    @staticmethod
    def convert_state(state=None, observation_space=None, *args, **kwargs):
        state, observation_space = obs_inventory_filter(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_toggle_equipped_items(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_resize(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_grayscale(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_pytorch_image(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space, state_buffer = obs_stack_image(state=state, observation_space=observation_space, *args, **kwargs)

        return state, observation_space, state_buffer 
    
    @staticmethod
    def create_action_set(functional_acts: bool = True, extracted_acts: bool = True):
        extracted_acts_filename = "extracted-actions.pickle"
        functional_acts_filename = "functional-actions.pickle"
        action_set = []

        if extracted_acts:
            e_filepath = os.path.join(minerl3161.actions_path, extracted_acts_filename)
            with open(e_filepath, "rb") as f:
                action_set.extend(pickle.load(f))

        if functional_acts:
            f_filepath = os.path.join(minerl3161.actions_path, functional_acts_filename)
            with open(f_filepath, "rb") as f:
                action_set.extend(pickle.load(f))
        
        return action_set

    @staticmethod
    def convert_action(action: int = None, action_space=None, action_set=None, last_unprocessed_state=None,):
        if action_space is not None:
            action_space = gym.spaces.Discrete(len(action_set))

        if action is not None:
            action = action_set[action]

            if action["place"] == "place_navigate":
                action = MineRLWrapper._get_navigate_block(action, last_unprocessed_state)

        return action, action_space
    
    @staticmethod
    def _get_navigate_block(action: Dict[str, str], last_unprocessed_obs) -> Dict[str, str]:
        navigate_blocks = ["dirt", "cobblestone", "stone"]

        for block in navigate_blocks:
            if last_unprocessed_obs["inventory"][block].item() > 0:
                action["place"] = block
                break
        
        if action["place"] == "place_navigate":
            action["place"] = "none"

        return action


def minerlWrapper(env, *args, **kwargs):
    """
    Parameters:
        features=None, 
        include_equipped_items=False, 
        resize_w=64, 
        resize_h=64, 
        img_feature_name="pov", 
        n_stack=4,
        functional_acts=True,
        extracted_acts=True,
    """
    return MineRLWrapper(env, *args, **kwargs)
