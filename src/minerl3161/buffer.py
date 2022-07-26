from collections import namedtuple
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_", "r", "d"])


class ReplayBuffer:
    """Stores experience for agent training. Currently assumes that actions are scalars
    """

    def __init__(self, n: int, state_shape: Tuple[int]) -> None:
        """Initialises a ReplayBuffer

        Args:
            n (int): size of ReplayBuffer
            state_shape (Tuple[int]): state shape to be stored
        """
        self.max_samples = n
        
        self.states = np.zeros((n, *state_shape), dtype=np.float32)
        self.actions = np.zeros((n, 1), dtype=np.float32)
        self.next_states = np.zeros((n, *state_shape), dtype=np.float32)
        self.rewards = np.zeros((n, 1), dtype=np.float32)
        self.dones = np.zeros((n, 1), dtype=bool)

        self.counter = 0
        self.full = False
    
    def add(self, 
                state: np.ndarray, 
                action: np.ndarray,
                next_state: np.ndarray, 
                reward: np.ndarray, 
                done: np.ndarray
            ) -> None:
        """adds a single timestep of experience to the experience buffer

        Args:
            state (np.ndarray): environment state
            action (np.ndarray): environment action
            next_state (np.ndarray): environment next state
            reward (np.ndarray): environment reward
            done (np.ndarray): environment done flag
        """
        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.next_states[self.counter] = next_state
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
        with open(save_path, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path: str):
        """Loads a ReplayBuffer from file

        Args:
            path (str): path to load from

        Returns:
            ReplayBuffer: loaded buffer
        """
        with open(path, 'rb') as infile:
            return pickle.load(infile)
    
    @staticmethod
    def load_from_paths(load_paths: List[str]):
        """Loads a replaybuffer from a list of paths

        Args:
            load_paths (List[str]): list of paths to load from
        """
        with open(load_paths[0], 'rb') as infile:
                buffer = pickle.load(infile)

        for path in load_paths[1:]:
            with open(path, 'rb') as infile:
                new_buffer = pickle.load(infile)

                assert buffer.state_shape == new_buffer.state_shape
                assert buffer.action_shape == new_buffer.action_shape

                buffer.memory += new_buffer.memory
                buffer.samples += new_buffer.samples
                buffer.max_samples += new_buffer.max_samples
    
    def join(self, b: 'ReplayBuffer'):
        """concatenation of b onto self

        Args:
            b (ReplayBuffer): other replay buffer to be added. Won't be modified by this function.
        """
        self.states = np.concatenate([self.states, b.states], axis=0)
        self.actions = np.concatenate([self.actions, b.actions], axis=0)
        self.next_states = np.concatenate([self.next_states, b.next_states], axis=0)
        self.rewards = np.concatenate([self.rewards, b.rewards], axis=0)
        self.dones = np.concatenate([self.dones, b.dones], axis=0)

        self.max_samples += b.max_samples

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.dones[idx]
    
    def sample(self, batch_size: int) -> Dict[np.ndarray]:
        raise NotImplementedError  # TODO
