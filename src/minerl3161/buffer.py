import os
import pickle
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np

Transition = namedtuple("Transition", ["s", "a", "s_", "r", "d"])


class ReplayBuffer:
    """Stores experience for agent training. Currently assumes that actions are scalars"""

    def __init__(self, n: int, obs_space: Dict[str, np.ndarray]) -> None:
        """Initialises a ReplayBuffer

        Args:
            n (int): size of ReplayBuffer
            obs_space (Dict[str, np.ndarray]): obs_space to be stored
        """
        self.max_samples = n

        try:
            self.feature_names = list(obs_space.keys())
        except AttributeError:
            self.feature_names = list(obs_space.spaces.keys())

        self.states = self._create_state_buffer(n, obs_space)
        self.actions = np.zeros((n, 1), dtype=np.float32)
        self.next_states = self._create_state_buffer(n, obs_space)
        self.rewards = np.zeros((n, 1), dtype=np.float32)
        self.dones = np.zeros((n, 1), dtype=bool)

        self.counter = 0
        self.full = False

    def _create_state_buffer(self, n, obs_space):
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
        """adds a single timestep of experience to the experience buffer

        Args:
            state (np.ndarray): environment state
            action (np.ndarray): environment action
            next_state (np.ndarray): environment next state
            reward (np.ndarray): environment reward
            done (np.ndarray): environment done flag
        """

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

    def __len__(self):
        return self.max_samples if self.full else self.counter

    def save(self, save_path):
        """Saves the current repla ybuffer

        Args:
            save_path (str): path to save to
        """
        with open(save_path, "wb") as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        """Loads a ReplayBuffer from file

        Args:
            path (str): path to load from

        Returns:
            ReplayBuffer: loaded buffer
        """
        with open(path, "rb") as infile:
            return pickle.load(infile)

    @staticmethod
    def load_from_paths(load_paths: List[str]):
        """Loads a replaybuffer from a list of paths

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

    def __getitem__(self, idx):
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

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:

        # create list of IDs to sample from our experience
        idxs = np.random.randint(
            low=0,
            high=self.counter if not self.full else self.max_samples,
            size=batch_size,
        )

        # we will return the sample in a dictionary
        batch_sample = {}
        batch_sample["rewards"] = self.rewards[idxs]
        batch_sample["dones"] = self.dones[idxs]
        batch_sample["actions"] = self.actions[idxs]
        # state and next state are dictionaries, so init them here and then fill them down below
        batch_sample["state"] = {}
        batch_sample["next_states"] = {}

        # fill in state and next state dictionaries
        for key in self.feature_names:
            batch_sample["state"][key] = self.states[key][idxs]
            batch_sample["next_states"][key] = self.next_states[key][idxs]

        return batch_sample
