import os

import gym
import minerl
import numpy as np
from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.utils import pt_dict_to_np, sample_pt_state
from minerl3161.wrappers import minerlWrapper



def test_dqnagent_dummy():
    state_space_shape = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }
    n_actions = 32
    hyperparams = DQNHyperparameters()
    hyperparams.feature_names = list(state_space_shape.keys())
    device = "cpu"
    save_path = "test.pt"

    agent = DQNAgent(
        obs_space=state_space_shape, 
        n_actions=n_actions, 
        device=device, 
        hyperparams=hyperparams)

    sample_state = sample_pt_state(state_space_shape, state_space_shape.keys())
    sample_state = pt_dict_to_np(sample_state)

    _ = agent.act(sample_state, train=True, step = 500)  # test epsilon greedy
    _ = agent.act(sample_state, train=False, step = None)  # test greedy act

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)


def test_dqnagent_full(minerl_env):
    wrapped_minerl_env = minerlWrapper(minerl_env)
    
    n_actions = 16

    hyperparams = DQNHyperparameters()
    device = "cpu"
    save_path = "test.pt"

    agent = DQNAgent(
        obs_space=wrapped_minerl_env.observation_space, 
        n_actions=n_actions, 
        device=device, 
        hyperparams=hyperparams)

    s = wrapped_minerl_env.reset()

    a1, _ = agent.act(s, train=True, step = 0)  # test epsilon greedy, random
    a2, _ = agent.act(s, train=True, step = 1e9)  # test epsilon greedy, greedy
    a3, _ = agent.act(s, train=False, step = None)  # test greedy act

    _ = wrapped_minerl_env.step(a1)
    _ = wrapped_minerl_env.step(a2)
    _ = wrapped_minerl_env.step(a3)

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)
