import numpy as np

from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import DQNHyperparameters


def test_dqnagent():
    state_space_shape = (3,)
    n_actions = 32
    hyperparams = DQNHyperparameters()

    agent = DQNAgent(state_space_shape, n_actions, hyperparams)

    sample_state = np.random.rand(state_space_shape)

    sample_action = agent.act(sample_state)