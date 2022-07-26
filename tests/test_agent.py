import os

import numpy as np
from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import DQNHyperparameters


def test_dqnagent():
    state_space_shape = (3,)
    n_actions = 32
    hyperparams = DQNHyperparameters()
    device = "cpu"
    save_path = "test.pt"

    agent = DQNAgent(state_space_shape, n_actions, device, hyperparams)

    sample_state = np.random.rand(state_space_shape[0]).astype(np.float32)

    sample_action = agent.act(sample_state)

    action = agent.eps_greedy_act(sample_state, 500)

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)
