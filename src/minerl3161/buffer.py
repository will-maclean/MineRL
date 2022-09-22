import os
import pickle
from collections import namedtuple
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import random

from minerl3161.segment_tree import MinSegmentTree, SumSegmentTree

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
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
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


    @staticmethod
    def create_batch_sample(rewards, dones, actions, states, next_states):
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

        # create list of IDs to sample from our experience
        idxs = np.random.randint(
            low=0,
            high=self.counter if not self.full else self.max_samples,
            size=batch_size,
        )

        return self.create_batch_sample(
            self.rewards[idxs],
            self.dones[idxs],
            self.actions[idxs],
            {key: self.states[key][idxs] for key in self.feature_names},
            {key: self.next_states[key][idxs] for key in self.feature_names})


class PrioritisedReplayBuffer(ReplayBuffer):

    def __init__(
        self, 
        n: int, 
        obs_space: Dict[str, np.ndarray], 
        alpha: float
    ) -> None:
        super(PrioritisedReplayBuffer, self).__init__(n, obs_space)

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_samples:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        super().add(state, action, next_state, reward, done)
    
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_samples

    def sample(
        self, 
        batch_size: int,
        beta: float
    ) -> Dict[str, np.ndarray]:
        
        indices = self._sample_proportional(batch_size)

        states = {}
        next_states = {}
        for k in self.states[0].keys():
            states[k] = self.states[k][indices]
            next_states[k] = self.next_states[k][indices]
        
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return ReplayBuffer.create_batch_sample(rewards, dones, actions, states, next_states), weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
