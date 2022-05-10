from collections import namedtuple
import os
import pickle

import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_", "r", "d"])


class ReplayBuffer:
    def __init__(self, n, state_shape) -> None:
        self.max_samples = n
        
        self.states = np.zeros((n, *state_shape), dtype=np.float32)
        self.actions = np.zeros((n, 1), dtype=np.float32)
        self.next_states = np.zeros((n, *state_shape), dtype=np.float32)
        self.rewards = np.zeros((n, 1), dtype=np.float32)
        self.dones = np.zeros((n, 1), dtype=bool)

        self.counter = 0
        self.full = False
    
    def add(self, state, action, next_state, reward, done):
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
        with open(save_path, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as infile:
            return pickle.load(infile)
    
    @staticmethod
    def load_from_paths(load_paths):
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
    
    def join(self, b):
        self.states = np.concatenate([self.states, b.states], axis=0)
        self.actions = np.concatenate([self.actions, b.actions], axis=0)
        self.next_states = np.concatenate([self.next_states, b.next_states], axis=0)
        self.rewards = np.concatenate([self.rewards, b.rewards], axis=0)
        self.dones = np.concatenate([self.dones, b.dones], axis=0)

        self.max_samples += b.max_samples

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.dones[idx]
