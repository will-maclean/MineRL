import torch as th
from torch import nn
import numpy as np
import gym
import minerl

from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models import DQNNet
from minerl3161.utils import np_dict_to_pt, sample_pt_state
from minerl3161.wrappers import mineRLObservationSpaceWrapper



def test_dqnnet_dummy():
    state_space = {
        'a': np.zeros(3), 
        'pov': np.zeros((4, 16, 16)),
        }
    
    n_actions = 32
    layer_size = 8
    batch_size = 3
    device = "cpu"

    net = DQNNet(state_space, n_actions, layer_size=layer_size).to(device)
    optim = th.optim.Adam(lr=0.01, params=net.parameters())

    sample_input = sample_pt_state(state_space, state_space.keys(), batch=1)
    sample_output = th.rand((batch_size, n_actions))

    nn_output = net(sample_input)

    loss = nn.functional.mse_loss(nn_output, sample_output)

    optim.zero_grad()
    loss.backward()
    optim.step()


def test_dqnnet_real(minerl_env):
    # real observation space
    w = 16
    h = 16
    features = ['pov', 'stone_sword', 'stonecutter', 'stone_shovel']
    stack = 4
    n_actions = 8
    device="cpu"
    layer_size = 8

    env = mineRLObservationSpaceWrapper(minerl_env, frame=stack, features=features, downsize_width=w, downsize_height=h)

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