import random
from typing import Dict, List

import numpy as np

from minerl3161.utils.segment_tree import SumSegmentTree, MinSegmentTree
from .replay_buffer import ReplayBuffer


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
    
    def __len__(self):
        return self.max_samples if self.full else self.tree_ptr

    def sample(
        self, 
        batch_size: int,
        beta: float
    ) -> Dict[str, np.ndarray]:
        
        indices = self._sample_proportional(batch_size)

        states = {}
        next_states = {}
        for k in self.feature_names:
            states[k] = self.states[k][indices]
            next_states[k] = self.next_states[k][indices]
        
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return ReplayBuffer.create_batch_sample(states, actions, next_states, rewards, dones), weights, indices

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