import dataclasses
import os

import gym
import minerl
import numpy as np
from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.utils import pt_dict_to_np, sample_pt_state
from minerl3161.wrappers import minerlWrapper


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


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
        hyperparams=hyperparams,
    )

    sample_state = sample_pt_state(state_space_shape, state_space_shape.keys())
    sample_state = pt_dict_to_np(sample_state)

    _ = agent.act(sample_state, train=True, step=500)  # test epsilon greedy
    _ = agent.act(sample_state, train=False, step=None)  # test greedy act

    # copy the weights of the two models across
    agent.q1.load_state_dict(agent.q2.state_dict())
    assert compare_models(agent.q1, agent.q2)  # this passes!

    # test the two models output the same stuff
    sample_state = sample_pt_state(state_space_shape, state_space_shape.keys(), batch=1)
    q1_out = agent.q1(sample_state)
    q1_out_2 = agent.q1(sample_state)
    q2_out = agent.q2(sample_state)

    assert q1_out.allclose(q1_out_2)  # check model pass is idempotent -> passes
    assert q1_out.allclose(q2_out)  # this also passes

    # test saving
    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)


def test_dqnagent_full(minerl_env):
    hyperparams = DQNHyperparameters()

    wrapped_minerl_env = minerlWrapper(minerl_env, **dataclasses.asdict(hyperparams))

    device = "cpu"
    save_path = "test.pt"

    agent = DQNAgent(
        obs_space=wrapped_minerl_env.observation_space, 
        n_actions=wrapped_minerl_env.action_space.n, 
        device=device, 
        hyperparams=hyperparams
        )

    s = wrapped_minerl_env.reset()

    a1, _ = agent.act(s, train=True, step=0)  # test epsilon greedy, random
    a2, _ = agent.act(s, train=True, step=1e9)  # test epsilon greedy, greedy
    a3, _ = agent.act(s, train=False, step=None)  # test greedy act

    _ = wrapped_minerl_env.step(a1)
    _ = wrapped_minerl_env.step(a2)
    _ = wrapped_minerl_env.step(a3)

    agent.save(save_path)

    del agent

    agent = DQNAgent.load(save_path)

    # clean up
    os.remove(save_path)
