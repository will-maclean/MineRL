from collections import namedtuple

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
        np.savez(save_path, s=self.states, a=self.actions, s_=self.next_states, r=self.rewards, d=self.dones)
    
    @staticmethod
    def load(load_path):
        load_dict = np.load(load_path)
        
        n = load_dict['s'].shape[0]
        shape = load_dict['s'].shape[1:]
        buffer = ReplayBuffer(n, shape)
        
        buffer.states = load_dict['s']
        buffer.actions = load_dict['a']
        buffer.next_states = load_dict['s_']
        buffer.rewards = load_dict['r']
        buffer.dones = load_dict['d']
        
        return buffer
    
    @staticmethod
    def load_from_paths(load_paths):
        if type(load_paths) == str:
            return ReplayBuffer.load(load_paths)
        elif len(load_paths) == 1:
            return ReplayBuffer.load(load_paths[0])
        else:
            buffer = ReplayBuffer.load(load_paths[0])

            for i in range(1, len(load_paths)):
                path = load_paths[i]
                new_buffer = ReplayBuffer.load(path)

                buffer.join(new_buffer)
            
            return buffer
    
    def join(self, b):
        self.states = np.concatenate([self.states, b.states], axis=0)
        self.actions = np.concatenate([self.actions, b.actions], axis=0)
        self.next_states = np.concatenate([self.next_states, b.next_states], axis=0)
        self.rewards = np.concatenate([self.rewards, b.rewards], axis=0)
        self.dones = np.concatenate([self.dones, b.dones], axis=0)

        self.max_samples += b.max_samples

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.dones[idx]
