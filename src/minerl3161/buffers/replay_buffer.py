from typing import Dict, List, Tuple, Union
from pathlib import Path
import pickle
from collections import namedtuple

import numpy as np

Transition = namedtuple("Transition", ["s", "a", "s_", "r", "d"])


class ReplayBuffer:
    """
    Stores experience for agent training. This class assumes that actions are scalar form.
    """

    def __init__(self, n: int, obs_space: Dict[str, np.ndarray]) -> None:
        """
        Initialises a ReplayBuffer

        Args:
            n (int): size of ReplayBuffer
            obs_space (Dict[str, np.ndarray]): obs_space to be stored
        """
        self.max_samples = n

        try:
            self.feature_names = list(obs_space.keys())
        except AttributeError:
            self.feature_names = list(obs_space.spaces.keys())
        
        self.obs_space = obs_space

        self.states = self._create_state_buffer(n, obs_space)
        self.actions = np.zeros((n, 1), dtype=np.float32)
        self.next_states = self._create_state_buffer(n, obs_space)
        self.rewards = np.zeros((n, 1), dtype=np.float32)
        self.dones = np.zeros((n, 1), dtype=bool)

        self.counter = 0
        self.full = False

    def _create_state_buffer(self, n, obs_space):
        """
        TODO
        """
        buf = {}

        for feature in self.feature_names:
            buf[feature] = np.zeros((n, *obs_space[feature].shape), dtype=np.float32)

        return buf

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Adds a single timestep of experience (transition) to the experience buffer

        Args:
            state (np.ndarray): the environment state at the given time step
            action (np.ndarray): the action taken in the envrionment at the given time step
            next_state (np.ndarray): the environment state the agent ends up in after taking the action
            reward (np.ndarray): the reward obtained from performing the action
            done (np.ndarray): a flag that represents whether or not the taken action ended the current episode
        """
        #TODO: is this comment needed?
        # this has potential to fail hard, but that's good - we want
        # the code to fail if the observation shape starts changing
        try:
            feature_names = state.keys()
        except AttributeError:
            feature_names = state.spaces.keys()

        for feature in feature_names:
            self.states[feature][self.counter] = state[feature]
            self.next_states[feature][self.counter] = next_state[feature]

        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done

        self.counter += 1

        if self.counter >= self.max_samples:
            self.full = True
            self.counter = 0

        return self.full

    def __len__(self) -> int:
        """
        Returns the length of the ReplayBuffer, or how many experience points are inside it

        Returns:
            int: the length of the ReplayBuffer, or how many experience points are inside it
        """
        return self.max_samples if self.full else self.counter

    def save(self, save_path: str) -> None:
        """
        Saves the current replay buffer

        Args:
            save_path (str): path to save to
        """
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'ReplayBuffer':
        """
        Loads a ReplayBuffer from file

        Args:
            path (str): path to load from

        Returns:
            ReplayBuffer: returns a loaded ReplayBuffer
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)

    @staticmethod
    def load_from_paths(load_paths: List[str]) -> None:
        """
        Loads a replay buffer from a list of paths

        Args:
            load_paths (List[str]): list of paths to load from
        """
        with open(load_paths[0], "rb") as infile:
            buffer = pickle.load(infile)

        for path in load_paths[1:]:
            with open(path, "rb") as infile:
                new_buffer = pickle.load(infile)

                assert buffer.state_shape == new_buffer.state_shape
                assert buffer.action_shape == new_buffer.action_shape

                buffer.memory += new_buffer.memory
                buffer.samples += new_buffer.samples
                buffer.max_samples += new_buffer.max_samples

    def __getitem__(self, idx: int) -> Tuple[dict, np.ndarray, dict, np.ndarray, np.ndarray]:
        """
        Retrieves a data point from the replay buffer at the supplied index

        Args:
            idx (int): the index of the item being retrieved from the buffer
        
        Returns:
            (dict, np.ndarray, dict, np.ndarray, np.ndarray): a tuple containing the experience from a given timestep
        """
        state = {}
        next_state = {}

        for key in self.feature_names:
            state[key] = self.states[key][idx]
            next_state[key] = self.next_states[key][idx]

        return (
            state,
            self.actions[idx],
            next_state,
            self.rewards[idx],
            self.dones[idx],
        )

    @staticmethod
    def create_batch_sample(
        states: np.ndarray, 
        actions: np.ndarray, 
        next_states: np.ndarray, 
        rewards: np.ndarray, 
        dones: np.ndarray
    ) -> Dict[str, Union[dict, np.ndarray]]:
        """
        Creates a batch of experience points from the supplied data, to be used for training the model

        Args:
            states (np.ndarray): a non-sequential sequence of states from the ReplayBuffer
            actions (np.ndarray): a non-sequential sequence of actions from the ReplayBuffer
            next_states (np.ndarray): a non-sequential sequence of next_states from the ReplayBuffer
            rewards (np.ndarray): a non-sequential sequence of rewards from the ReplayBuffer
            dones (np.ndarray): a non-sequential sequence of dones from the ReplayBuffer
        
        Returns:
            Dict[str, Union[dict, np.ndarray]]: a dictionary where each value is the batch of data passed into the method
        """
        # return the sample in a dictionary
        batch_sample = {}
        batch_sample["reward"] = rewards
        batch_sample["done"] = dones
        batch_sample["action"] = actions
        # state and next state are dictionaries, so init them here and then fill them down below
        batch_sample["state"] = {}
        batch_sample["next_state"] = {}

        # fill in state and next state dictionaries
        for key in states:
            batch_sample["state"][key] = states[key]
            batch_sample["next_state"][key] = next_states[key]

        return batch_sample


    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample method used to retrieve a batch of experience data points

        Args:
            batch_size (int): the size of the batch - how many experience points the batch should contain
        
        Returns:
            Dict[str, np.ndarray]: a dictionary which contains the data points
        """
        # create list of IDs to sample from our experience
        idxs = np.random.randint(
            low=0,
            high=self.counter if not self.full else self.max_samples,
            size=batch_size,
        )

        return self.create_batch_sample(
            {key: self.states[key][idxs] for key in self.feature_names},
            self.actions[idxs],
            {key: self.next_states[key][idxs] for key in self.feature_names},
            self.rewards[idxs],
            self.dones[idxs],
        )