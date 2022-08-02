import pytest

from copy import deepcopy
from minerl3161.utils import linear_decay, epsilon_decay, copy_weights, sample_pt_state
from minerl3161.models import DQNNet
import numpy as np
import torch


def nn_params_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_linear_decay():
    out = linear_decay(5, 1.0, 0.0, 10)
    assert out == 0.5

    out = linear_decay(11, 1.0, 0.0, 10)
    assert out == 0.0


def test_epsilon_decay():
    # epsilon decay is harder to test

    out = epsilon_decay(5, 1.0, 0.1, 10)
    assert 0.1 <= out < 1.0

    out = epsilon_decay(5000000000000, 1.0, 0.1, 10)
    assert 0.1 <= out < 1.0


def test_copy_weights():
    # hard copy
    n1 = DQNNet((5,), 5, 5)
    n2 = DQNNet((5,), 5, 5)

    assert not nn_params_equal(n1, n2)

    copy_weights(n1, n2)

    assert nn_params_equal(n1, n2)

    # soft copy
    n1 = DQNNet((5,), 5, 5)
    n2 = DQNNet((5,), 5, 5)

    n1_copy = deepcopy(n1)
    n2_copy = deepcopy(n2)

    assert not nn_params_equal(n1, n2)

    copy_weights(n1, n2, 0.5)

    assert nn_params_equal(n1, n1_copy)  # n1 should not have changed
    assert not nn_params_equal(n2, n2_copy)  # n2 should have changed


def test_sample_pt_state():
    sample_observation_space = {
        "f1": np.zeros(3, 64, 64),
        "f2": np.zeros(4),
        "f2": np.zeros(6),
    }

    feature_names = ["f1", "f2"]

    sample = sample_pt_state(sample_observation_space, feature_names)

    assert type(sample) == dict
    assert type(sample["f1"]) == torch.tensor
    assert type(sample["f2"]) == torch.tensor
    assert sample["f1"].shape == (3, 64, 64)
    assert sample["f2"].shape == (4)

    with pytest.raises(KeyError):
        sample["f3"]
