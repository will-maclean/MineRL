import random
from typing import Dict, List

import numpy as np

from minerl3161.utils.segment_tree import SumSegmentTree, MinSegmentTree
from .replay_buffer import ReplayBuffer


class PrioritisedReplayBuffer(ReplayBuffer):
    """
    Implements the Prioritised Experience Replay algorihtm, and inherits from the ReplayBuffer class. The ReplayBuffer class randomly samples
    from it's buffers when producing a batch. This is improved upon in this class, where the experience points used to train the model are 
    selected based on the magnitude of the loss associated with that experience point, or rather, how much value that experience point contributes
    to the model's learning.
    """

    def __init__(
        self, 
        n: int, 
        obs_space: Dict[str, np.ndarray], 
        alpha: float
    ) -> None:
        """
        Initialises a PrioritisedReplayBuffer

        Args:
            n (int): size of ReplayBuffer
            obs_space (Dict[str, np.ndarray]): obs_space to be stored
            alpha (float): a value ranging between 0 and 1 where 0 means no prioritisation is used, and 1 means full prioritisation is used
        """
        super(PrioritisedReplayBuffer, self).__init__(n, obs_space)

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # Capacity must be positive and a power of 2.
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
        """
        Adds a single timestep of experience (transition) to the experience buffer

        Args:
            state (np.ndarray): the environment state at the given time step
            action (np.ndarray): the action taken in the envrionment at the given time step
            next_state (np.ndarray): the environment state the agent ends up in after taking the action
            reward (np.ndarray): the reward obtained from performing the action
            done (np.ndarray): a flag that represents whether or not the taken action ended the current episode
        """
        super().add(state, action, next_state, reward, done)
    
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_samples
    
    def __len__(self):
        """
        Returns the length of the ReplayBuffer, or how many experience points are inside it

        Returns:
            int: the length of the ReplayBuffer, or how many experience points are inside it
        """
        return self.max_samples if self.full else self.tree_ptr

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        """
        Sample method used to retrieve a batch of experience data points. Is dependent on bata which is used to preserve the uniform distribution
        obtained when randomly sampling in a normal Replay Buffer. This process is known as Importance Sampling.

        Args:
            batch_size (int): the size of the batch - how many experience points the batch should contain
            beta (float): a value between 0 and 1, that anneals from a starting value to a final value (typically 1), controls how the
                          prioritisation is being applied
        
        Returns:
            Dict[str, np.ndarray]: a dictionary which contains the data points
        """
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

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Used to update the priorities (it's value for learning) associated with a given experience point. Is performed on a set
        of experience points, donated by the list of indicies

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            indices (List[int]): the list of indices whose priorities should be updated
            priorities (np.ndarray): the list of priorities to be updated
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self, batch_size: int) -> List[int]:
        """
        Used to sample the indicies based on the proportions as stored in the segment tree. The retrieved indicies correspond
        to the priorities that are to be updated.

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            batch_size (int): the number of indicies to be retrieved from the segment tree
        
        Returns:
            List[int]: a list containing the indicies of the priorities that need to be updated in the buffer
        """
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
        """
        Used to calculate the weight of the experience point at the supplied index. This determines how often the model should see
        this specific transition with respect to the magnitude of the loss obtained from this sample.

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            idx (int): the index of the transition whose weight is being calculated
            beta (float): a value that anneals towards beta_max, which is used as apart of Importance Sampling, to ensure the weights,
                          (which determines the frequency the experience is fed into the model) preserves a uniform distribution to 
                          prevent unwanted bais
        
        Returns:
            float: the calculated weight corresponding to the transition at the specified index
        """
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weight
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight