import os

import gym
import minerl
import numpy as np
from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.utils import pt_dict_to_np, sample_pt_state
from minerl3161.wrappers import mineRLObservationSpaceWrapper



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

    _ = agent.act(sample_state)

    _ = agent.eps_greedy_act(sample_state, 500)

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)


def test_dqnagent_full(minerl_env):

    w = 16
    h = 16
    stack = 4
    n_actions = 16

    hyperparams = DQNHyperparameters()
    env = mineRLObservationSpaceWrapper(minerl_env, frame=stack, features=hyperparams.inventory_feature_names, downsize_width=w, downsize_height=h)
    device = "cpu"
    save_path = "test.pt"

    agent = DQNAgent(
        obs_space=env.observation_space, 
        n_actions=n_actions, 
        device=device, 
        hyperparams=hyperparams)

    sample_state = sample_pt_state(env.observation_space, hyperparams.feature_names)
    sample_state = pt_dict_to_np(sample_state)

    _ = agent.act(sample_state)

    _ = agent.eps_greedy_act(sample_state, 500)

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)
