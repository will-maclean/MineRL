import torch as th
from torch import nn

from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.models import DQNNet


def test_dqnnet():
    state_space_shape = (3,)
    n_actions = 32
    layer_size = 8
    batch_size = 3
    device = "cpu"

    net = DQNNet(state_space_shape, n_actions, layer_size).to(device)
    optim = th.optim.Adam(lr=0.01, params=net.parameters())

    sample_input = th.rand((batch_size, *state_space_shape), device=device)
    sample_output = th.rand((batch_size, n_actions))

    nn_output = net(sample_input)

    loss = nn.functional.mse_loss(nn_output, sample_output)

    optim.zero_grad()
    loss.backward()
    optim.step()
