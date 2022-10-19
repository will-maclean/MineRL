from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pickle

import numpy as np
import pandas as pd
import gym
import cv2

import minerl3161

def decode_action(obj: dict, camera_shrink_factor: int = 100) -> dict:
    """
    Decodes an action to fit into a dataframe.
    Helper function for MineRLWrapper.map_action()

    Args:
        obj (dict): action to be decoded
        camera_shrink_factor (int): the factor to reduce the camera deltas by
    
    Returns:
        dict: the decoded action 
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


def obs_grayscale(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, gym.Space] = None, img_feature_name: str = 'pov', *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Applies grayscale effect to an image feature in an observation

    Args:
        state (Dict[str, np.ndarray], optional): state to apply effect to. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to be modified. Defaults to None.
        img_feature_name (str, optional): name of image feature. Defaults to 'pov'.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified action, modified observation space
    """

    if observation_space is not None:
        observation_space.spaces[img_feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*observation_space[img_feature_name].shape[:-1], 1)
        )
    
    if state is not None:
        rgb_weights = np.array([0.2989, 0.5870, 0.1140])
        intermediate = state[img_feature_name] * rgb_weights
        state[img_feature_name] = np.expand_dims(np.sum(intermediate, axis = 2), 2)

    return state, observation_space


def obs_resize(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, np.ndarray] = None, 
                img_feature_name: str = 'pov', resize_w: int = 64, resize_h: int = 64, *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Resizes an image observation in a state

    Args:
        state (Dict[str, np.ndarray], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modify. Defaults to None.
        img_feature_name (str, optional): name of image feature. Defaults to 'pov'.
        resize_w (int, optional): width to resize to. Defaults to 64.
        resize_h (int, optional): height to resize to. Defaults to 64.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified state, modified observation space
    """
    if observation_space is not None:
        observation_space.spaces[img_feature_name] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(resize_w, resize_h, 3),
            dtype=observation_space[img_feature_name].dtype,
        )
    
    if state is not None:
        state[img_feature_name] = cv2.resize(state[img_feature_name], (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    
    return state, observation_space


def obs_pytorch_image(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, np.ndarray] = None, img_feature_name: str = 'pov', *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Modifies an image in np format (W, H, C) to be in PyTorch format (C, W, H)

    Args:
        state (Dict[str, np.ndarray], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modify. Defaults to None.
        img_feature_name (str, optional): name of image feature. Defaults to 'pov'.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified state, modified observation space
    """
    if observation_space is not None:
        observation_space.spaces[img_feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(observation_space[img_feature_name].shape[2], observation_space[img_feature_name].shape[1], observation_space[img_feature_name].shape[0])
        )

    if state is not None:
        state[img_feature_name] = np.swapaxes(state[img_feature_name], 0, 2) / 255
    
    return state, observation_space


def obs_stack_image(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, np.ndarray] = None, img_feature_name: str = 'pov', 
                    state_buffer: Union[np.ndarray, None] = None, n_stack: int = 4, *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Stacks last n_stack images into a single image observation

    Args:
        state (Dict[str, np.ndarray], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modify. Defaults to None.
        img_feature_name (str, optional): name of image feature. Defaults to 'pov'.
        state_buffer (Union[np.ndarray, None], optional): buffer of the last n_stack observations. Defaults to None.
        n_stack (int, optional): hot many images to stack. Defaults to 4.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: mopdified state, modified observation space
    """
    if observation_space is not None:
        observation_space.spaces[img_feature_name] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(n_stack, observation_space[img_feature_name].shape[1], observation_space[img_feature_name].shape[2])
        )
        if state_buffer is None:
            state_buffer = np.zeros((n_stack, *observation_space[img_feature_name].shape[1:]))

    if state is not None:
        if state_buffer is None:
            state_buffer = np.zeros((n_stack, *state[img_feature_name].shape[1:]))

        state_buffer = np.roll(state_buffer, shift=-1, axis=0)
        new_state = state[img_feature_name]
        state_buffer[-1] = np.squeeze(new_state, axis=0)

        state[img_feature_name] = state_buffer.copy()

    return state, observation_space, state_buffer


def obs_inventory_filter(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, np.ndarray] = None, inventory_feature_names: Union[List[str], None] = None, 
                        inv_feature_max: int = 16, *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Processes an inventory observation to be a single vector, and also appled clipping and scaling

    Args:
        state (Dict[str, np.ndarray], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modfy. Defaults to None.
        inventory_feature_names (Union[List[str], None], optional): name of inventory featuer to include. Set to ["all"] to use all features. Defaults to None.
        inv_feature_max (int, optional): max value to clip to. Defaults to 16.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified state, modified observation space
    """
    if observation_space is not None:
        if len(inventory_feature_names) == 1 and inventory_feature_names[0] == "all":
            # return all the items
            observation_space.spaces['inventory'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(len(list(observation_space['inventory'].spaces.keys())),)
            )
        elif len(inventory_feature_names) > 0 and "all" not in inventory_feature_names:
            # return a specified set of features
            observation_space.spaces['inventory'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(len(inventory_feature_names),)
            )
        else:
            raise ValueError(f"features must be either ['all'] or a list of features not containing 'all'. Features is: {inventory_feature_names}")

    if state is not None:
        inventory = []

        if len(inventory_feature_names) == 1 and inventory_feature_names[0] == "all":
            # return all the items
            for key in state['inventory']:
                inventory.append(
                    min(state['inventory'][key], inv_feature_max) / inv_feature_max
                    )
            
        elif len(inventory_feature_names) > 0 and "all" not in inventory_feature_names:
            # return a specified set of features
            for key in inventory_feature_names:
                inventory.append(
                    min(state['inventory'][key], inv_feature_max) / inv_feature_max
                    )
            
        else:
            raise ValueError(f"features must be either ['all'] or a list of features not containing 'all'. Features is: {inventory_feature_names}")
        
        state['inventory'] = np.array(inventory)

    return state, observation_space


def obs_toggle_equipped_items(state: Optional[Dict[str, np.ndarray]] = None, observation_space: Dict[str, np.ndarray] = None, 
                                include_equipped_items: Optional[List[str]] = False, *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Controls whether the equipped_items feature should be included in the observation

    Args:
        state (Optional[Dict[str, np.ndarray]], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modify. Defaults to None.
        include_equipped_items (Optional[List[str]], optional): whether to include equipped_items. Defaults to False.

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified state, modified actions
    """
    if observation_space is not None:
        if not include_equipped_items:
            # we need to make a copy of the observation space and use that instead so that we don't modify the original
            # observation space, as deleting is in-place
            del_keys = []
            for k in observation_space.spaces.keys():
                if k.startswith("equipped"):
                    del_keys.append(k)
            
            for k in del_keys:
                del observation_space.spaces[k]

    if state is not None:
        if not include_equipped_items:
            del_keys = []
            for k in state.keys():
                if k.startswith("equipped"):
                    del_keys.append(k)
            
            for k in del_keys:
                del state[k]

    return state, observation_space


def obs_compass(state: Dict[str, np.ndarray] = None, observation_space: Dict[str, np.ndarray] = None, 
                compass_name: str = "compass", *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]:
    """Processes compass observation by converting to np.ndarray and applying scaling

    Args:
        state (Dict[str, np.ndarray], optional): state to modify. Defaults to None.
        observation_space (Dict[str, np.ndarray], optional): observation space to modify. Defaults to None.
        compass_name (str, optional): name of compass feature. Defaults to "compass".

    Returns:
        Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, gym.Space]]]: modified state, modified observation space
    """
    if observation_space is not None:
        if "compass" in observation_space.spaces.keys():
            observation_space.spaces[compass_name] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,)
            )
        elif "compassAngle" in observation_space.spaces.keys():
            observation_space.spaces[compass_name] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,)
            )

            del observation_space.spaces["compassAngle"]
        
    if state is not None:
        # try:
        #     state[compass_name] = np.atleast_1d(state[compass_name]["angle"] / 180)
        # except KeyError:
        #     pass
        
        if "compass" in state.keys():
            if type(state[compass_name]) == np.ndarray:
                state[compass_name] = np.atleast_1d(state[compass_name] / 180)
            else:
                state[compass_name] = np.atleast_1d(state[compass_name]["angle"] / 180)
        
        elif "compassAngle" in state.keys():
            state[compass_name] = np.atleast_1d(state["compassAngle"] / 180)

            del state["compassAngle"]
    
    return state, observation_space



class MineRLWrapper(gym.Wrapper):
    """Wraps a MineRL environment. Can handle all MineRL environments.
    """

    def __init__(self, 
                env: gym.Env, 
                extracted_acts_filename: str = "extracted-actions.pickle", 
                functional_acts_filename: str = "functional-actions.pickle",
                functional_acts: bool = True,
                extracted_acts: bool = True,
                inventory_feature_names: Optional[List[str]] = None, 
                include_equipped_items: bool = False, 
                resize_w: int = 64, 
                resize_h: int = 64, 
                img_feature_name: str = "pov", 
                n_stack: int = 4,
                repeat_action: int = 1,
                *args,
                **kwargs,
        ) -> None:
        """Constructor

        Args:
            env (gym.Env): env to wrap
            extracted_acts_filename (str, optional): path to extracted actions pickle. Defaults to "extracted-actions.pickle".
            functional_acts_filename (str, optional): path to function actions pickle. Defaults to "functional-actions.pickle".
            functional_acts (bool, optional): whether to use functional actions. Defaults to True.
            extracted_acts (bool, optional): whether to use extracted actions. Defaults to True.
            inventory_feature_names (Optional[List[str]], optional): inventory features to include. Defaults to None.
            include_equipped_items (bool, optional): whether to use equiped_items feature. Defaults to False.
            resize_w (int, optional): resize width for image scaling. Defaults to 64.
            resize_h (int, optional): resize height for image scaling. Defaults to 64.
            img_feature_name (str, optional): name of image feature. Defaults to "pov".
            n_stack (int, optional): stack last n images. Defaults to 4.
            repeat_action (int, optional): repeat the specified action n times. Defaults to 1.
        """
        super().__init__(env)

        self.repeat_action = repeat_action

        self.obs_kwargs = {
            "inventory_feature_names": inventory_feature_names if inventory_feature_names is not None else ["all"],
            "resize_w": resize_w,
            "resize_h": resize_h,
            "img_feature_name": img_feature_name,
            "n_stack": n_stack,
            "include_equipped_items": include_equipped_items,
            "state_buffer": np.zeros((n_stack, resize_h, resize_w), dtype=np.float32),
            "last_unprocessed_state": None,
        }

        # update action space
        self.action_set = MineRLWrapper.create_action_set(
            functional_acts=functional_acts, 
            extracted_acts=extracted_acts, 
            functional_acts_filename=functional_acts_filename, 
            extracted_acts_filename=extracted_acts_filename
        )
        _, self.action_space = MineRLWrapper.convert_action(action_space=self.action_space, action_set=self.action_set)

        # update observation space
        _, self.observation_space, _ = MineRLWrapper.convert_state(observation_space=deepcopy(self.observation_space), **self.obs_kwargs)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment

        Returns:
            Dict[str, np.ndarray]: processed state
        """
        self.obs_kwargs["state_buffer"] = np.zeros_like(self.obs_kwargs["state_buffer"])
        self.obs_kwargs["last_unprocessed_state"] = self.env.reset()
        state, _, self.obs_kwargs["state_buffer"] = MineRLWrapper.convert_state(state=self.obs_kwargs["last_unprocessed_state"], **self.obs_kwargs)
        return state
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step environment

        Args:
            action (int): action for environment

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]: next state, reward, done, info
        """
        action, _ = MineRLWrapper.convert_action(action=action, last_unprocessed_state=self.obs_kwargs["last_unprocessed_state"], action_set=self.action_set)

        reward_sum = 0

        for _ in range(self.repeat_action):
            self.obs_kwargs["last_unprocessed_state"], reward, done, info = self.env.step(action)
            reward_sum += reward

            if done:
                break
        
        state, _, self.obs_kwargs["state_buffer"] = MineRLWrapper.convert_state(state=self.obs_kwargs["last_unprocessed_state"], **self.obs_kwargs)

        return state, reward_sum, done, info

    @staticmethod
    def map_action(obs:Dict[str, np.ndarray], action_set: dict) -> int:
        """ Maps an observation from the env/dataset to an action index in our action set
        Args:
            obs (dict): A single action from the env/dataset in dictionary form
            action_set (dict): The action set initialised by the MineRLWrapper
        """
        obs = decode_action(obs)
        cluster_centers = pd.DataFrame([decode_action(i) for i in action_set])

        # First checks if the action is categorical in nature
        cat_list = ['place', 'nearbyCraft', 'nearbySmelt', 'craft', 'equip']
        for cat_act in cat_list:
            if obs[cat_act] != 'none':
                mapped_action = cluster_centers[cluster_centers[cat_act] == obs[cat_act]].index
                if len(mapped_action) > 0:
                    return mapped_action[0]
        # The values of each numerical field in a list
        obs_num = list({k: v for k, v in obs.items() if k not in cat_list}.values())

        # Calculates the euclidean distance between `obs` and every action in action set
        distances = [
            np.linalg.norm(obs_num - action.values) for _, action in cluster_centers.drop(
                cat_list, axis=1).iterrows()
        ]
        return np.argmin(distances)

    @staticmethod
    def convert_state(state: Optional[Dict[str, np.ndarray]] = None, observation_space: Optional[Dict[str, np.ndarray]] = None, *args, **kwargs) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[gym.Space]]:
        """Converts state/observation space

        Args:
            state (Optional[Dict[str, np.ndarray]], optional): state to modify. Defaults to None.
            observation_space (Optional[Dict[str, np.ndarray]], optional): observation space to modify. Defaults to None.

        Returns:
            Tuple[Optional[Dict[str, np.ndarray]], Optional[gym.Space]]: modified state, modified observation space
        """
        state, observation_space = obs_compass(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_inventory_filter(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_toggle_equipped_items(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_resize(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_grayscale(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space = obs_pytorch_image(state=state, observation_space=observation_space, *args, **kwargs)
        state, observation_space, state_buffer = obs_stack_image(state=state, observation_space=observation_space, *args, **kwargs)

        return state, observation_space, state_buffer 
    
    @staticmethod
    def create_action_set(
        functional_acts: bool, 
        extracted_acts: bool, 
        extracted_acts_filename: str, 
        functional_acts_filename: str
    ) -> Dict[str, Any]:
        """Creates action space for suppled action pickles

        Args:
            functional_acts (bool): whether to use functional actions
            extracted_acts (bool): whether to use extracted actions
            extracted_acts_filename (str): path to extracted actions
            functional_acts_filename (str): path to functional actions

        Returns:
            Dict[str, Any]: created action set
        """
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
    def convert_action(action: Optional[int] = None, action_space: Any = None, 
                        action_set: Optional[dict]=None, last_unprocessed_state: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[gym.Space]]:
        """converts an action/action space

        Args:
            action (Optional[int], optional): action to modify. Defaults to None.
            action_space (Any, optional): action space to modify. Defaults to None.
            action_set (Optional[dict], optional): action set to use. Defaults to None.
            last_unprocessed_state (Optional[Dict[str, np.ndarray]], optional): last unprocessed state is required for some transforms. Defaults to None.

        Returns:
            Tuple[Optional[Dict[str, np.ndarray]], Optional[gym.Space]]: converted action, converted action space
        """
        if action_space is not None:
            action_space = gym.spaces.Discrete(len(action_set))

        if action is not None:
            action = action_set[action]

            if action["place"] == "place_navigate":
                action = MineRLWrapper._get_navigate_block(action, last_unprocessed_state)

        return action, action_space
    
    @staticmethod
    def _get_navigate_block(action: Dict[str, str], last_unprocessed_obs: Dict[str, np.ndarray]) -> Dict[str, str]:
        """utility function to determine which block should be placed, based on last state

        Args:
            action (Dict[str, str]): action to modify
            last_unprocessed_obs (Dict[str, np.ndarray]): last env observation

        Returns:
            Dict[str, str]: modified action
        """
        navigate_blocks = ["dirt", "cobblestone", "stone"]

        for block in navigate_blocks:
            if last_unprocessed_obs["inventory"][block].item() > 0:
                action["place"] = block
                break
        
        if action["place"] == "place_navigate":
            action["place"] = "none"

        return action


<<<<<<< HEAD:src/minerl3161/utils/wrappers.py
class CartpoleWrapper(gym.ObservationWrapper):
    """Provides a simple wrapper for CartPole-like environments to be used with our code.
    """
    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env)
        self.observation_space = {"state": self.observation_space}
    
    def observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Modifies observation to be a dictionary

        Args:
            observation (np.ndarray): origional observation

        Returns:
            Dict[str, np.ndarray]: modified observation
        """
        return {"state": observation}


def cartPoleWrapper(env: gym.Env, *args, **kwargs) -> CartpoleWrapper:
    """Creates the cartpole wrapper

    Args:
        env (gym.Env): env to wrap

    Returns:
        CartpoleWrapper: wrapped env
=======
def minerlWrapper(env, *args, **kwargs):
>>>>>>> 7e73f64bf6f1381fe39076b2e53fc89833056687:src/minerl3161/wrappers/minerl_wrapper.py
    """
    return CartpoleWrapper(env, *args, **kwargs)

def minerlWrapper(env: gym.env, *args, **kwargs) -> MineRLWrapper:
    """Wraps a MineRL environment in our wrappers

    Args:
        env (gym.env): env to wrap

    Returns:
        MineRLWrapper: wrapped env
    """
    return MineRLWrapper(env, *args, **kwargs)