from collections import deque
from typing import Dict, Tuple

import numpy as np

from minerl3161.hyperparameters.rainbow_dqn_hp import RainbowDQNHyperparameters

from .replay_buffer import ReplayBuffer


class NStepReplayBuffer:
    """
    Implements the N Step Learning algorithm into a ReplayBuffer. A special buffer is required for this algorithm as n-step transitions
    are stored into the buffer. Subsequentially, this class does not inherit from the ReplayBuffer class.
    """

    def __init__(
        self,
        n: int, 
        hyperparameters: RainbowDQNHyperparameters
    ) -> None:
        """
        Initialises a NStepReplayBuffer

        Args:
            n (int): size of ReplayBuffer
            hyperparameters (RainbowDQNHyperparameters): the hyperparameters that are used internally in this class 
        """
        self.size = n
        self.batch_size = hyperparameters.batch_size

        self.n_step_buffer = deque(maxlen=hyperparameters.n_step)
        self.n_step = hyperparameters.n_step
        self.gamma = hyperparameters.gamma
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        This method adds the transition to the buffer, and unrolls the n-step data
        
        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Args:
            state (Dict[str, np.ndarray]): the environment state at the given time step
            action (Union[np.ndarray, float]): the action taken in the envrionment at the given time step
            next_state (Dict[str, np.ndarray]): the environment state the agent ends up in after taking the action
            reward (Union[np.ndarray, float]): the reward obtained from performing the action
            done (Union[np.ndarray, bool]): a flag that represents whether or not the taken action ended the current episode
        """
        transition = (state, action, next_state, reward, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        next_state, reward, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        return self[0]
    
    def __getitem__(self, idx: int):
        """
        Retrieves a data point from the buffer at the supplied index

        Args:
            idx (int): the index of the item being retrieved from the buffer
        
        Returns:
            (dict, np.ndarray, dict, np.ndarray, np.ndarray): a tuple containing the experience from a given timestep
        """
        return self.n_step_buffer[idx]
    
    def __len__(self):
        """
        Returns the length of the NStepReplayBuffer, or how many experience points are inside it

        Returns:
            int: the length of the NStepReplayBuffer, or how many experience points are inside it
        """
        return len(self.n_step_buffer)

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Sample method used to retrieve a batch of experience data points, from internally generated indices
        
        Returns:
            Dict[str, np.ndarray]: a dictionary which contains the data points from the generated indices
        """
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return NStepReplayBuffer.sample_batch_from_idxs(self, indices), indices

    @staticmethod
    def sample_batch_from_idxs(buffer: ReplayBuffer, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Creates a batch of experience points from a supplied ReplayBuffer and indices

        Args:
            buffer (ReplayBuffer): the replay buffer whose experience points are being used
            indices (np.ndarray): the set of integers that corresponds to the experience points to be sampled
        
        Returns:
            Dict[str, Union[dict, np.ndarray]]: a dictionary where each value is the batch of data passed into the method
        """
        states = {}
        next_states = {}
        for k in buffer.feature_names:
            states[k] = buffer.states[k][indices]
            next_states[k] = buffer.next_states[k][indices]
        
        actions = buffer.actions[indices]
        rewards = buffer.rewards[indices]
        dones = buffer.dones[indices]

        return ReplayBuffer.create_batch_sample(states, actions, next_states, rewards, dones)

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        """
        Calculates the n step values for the next_state, reward, and done values

        Adapted from Curt-Park: https://github.com/Curt-Park/rainbow-is-all-you-need

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool]: the calculated next_state, reward and done values after n steps
        """
        # info of the last transition
        next_state, reward, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            n_s, r, d = transition[-3:]

            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return next_state, reward, done