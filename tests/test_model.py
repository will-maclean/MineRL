import torch as th
from torch import nn
import numpy as np
import gym
import minerl

from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models.models import DQNNet
from minerl3161.utils.utils import np_dict_to_pt, sample_pt_state
from minerl3161.utils.wrappers import minerlWrapper


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_dqnnet_dummy():
    state_space = {
        "a": np.zeros(3),
        "pov": np.zeros((4, 16, 16)),
    }

    n_actions = 32
    layer_size = 8
    batch_size = 3
    device = "cpu"

    net = DQNNet(state_space, n_actions, layer_size=layer_size).to(device)
    optim = th.optim.Adam(lr=0.01, params=net.parameters())

    sample_input = sample_pt_state(state_space, state_space.keys(), batch=batch_size)
    sample_output = th.rand((batch_size, n_actions))

    nn_output = net(sample_input)

    loss = nn.functional.mse_loss(nn_output, sample_output)

    optim.zero_grad()
    loss.backward()
    optim.step()

    net2 = DQNNet(state_space, n_actions, layer_size=layer_size).to(device)
    net2.requires_grad_(False)
    net.load_state_dict(net2.state_dict())
    assert compare_models(net, net2)  # this passes!

    # test the two models output the same stuff
    sample_state = sample_pt_state(state_space, state_space.keys(), batch=1)
    q1_out = net(sample_state)
    q1_out_2 = net(sample_state)
    q2_out = net2(sample_state)

    assert q1_out.allclose(q1_out_2)  # check model pass is idempotent -> passes
    assert q1_out.allclose(q2_out)  # this fails!


def test_dqnnet_real(minerl_env):
    # real observation space
    w = 16
    h = 16
    inventory_feature_names = ['stone_sword', 'stonecutter', 'stone_shovel']
    n_stack = 4
    n_actions = 8
    device = "cpu"
    layer_size = 8

    env = minerlWrapper(minerl_env, n_stack=n_stack, inventory_feature_names=inventory_feature_names, resize_w=w, resize_h=h)

    obs_space = env.observation_space

    net = DQNNet(obs_space, n_actions, layer_size=layer_size).to(device)
    optim = th.optim.Adam(lr=0.01, params=net.parameters())

    sample_input = sample_pt_state(obs_space, obs_space.spaces.keys(), batch=1)
    sample_output = th.rand((1, n_actions))

    nn_output = net(sample_input)

    loss = nn.functional.mse_loss(nn_output, sample_output)

    optim.zero_grad()
    loss.backward()
    optim.step()
