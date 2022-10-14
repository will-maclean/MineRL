from collections import deque
from typing import Dict, Tuple

import numpy as np

from .replay_buffer import ReplayBuffer


class NStepReplayBuffer:

    def __init__(
        self,
        size: int, 
        batch_size: int, 
        gamma: float,
        n_step: int, 
    ) -> None:

        self.size = size
        self.batch_size = batch_size

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        transition = (state, action, next_state, reward, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        next_state, reward, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        return self[0]
    
    def __getitem__(self, idx):
        return self.n_step_buffer[idx]
    
    def __len__(self):
        return len(self.n_step_buffer)

    @staticmethod
    def sample_batch_from_idxs(buffer: ReplayBuffer, indices: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        states = {}
        next_states = {}
        for k in buffer.feature_names:
            states[k] = buffer.states[k][indices]
            next_states[k] = buffer.next_states[k][indices]
        
        actions = buffer.actions[indices]
        rewards = buffer.rewards[indices]
        dones = buffer.dones[indices]

        return ReplayBuffer.create_batch_sample(states, actions, next_states, rewards, dones)
    
    def sample(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return NStepReplayBuffer.sample_batch_from_idxs(self, indices), indices
    
    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step reward, next_state, and done."""
        # info of the last transition
        next_state, reward, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            n_s, r, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return next_state, reward, done