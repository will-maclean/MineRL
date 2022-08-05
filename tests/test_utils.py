from copy import deepcopy

import numpy as np
import torch as th

from minerl3161.utils import linear_decay, epsilon_decay, copy_weights, np_dict_to_pt
from minerl3161.models import DQNNet


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


def test_np_dict_to_pt():
    test_dict = {"a1": np.zeros(4), "a2": np.ones((3, 4))}

    converted_dict = np_dict_to_pt(test_dict)

    assert converted_dict["a1"] == th.zeros(4)
    assert converted_dict["a2"] == th.zeros((3, 4))


def test_pt_dict_to_np():
    test_dict = {"a1": th.zeros(4), "a2": th.ones((3, 4))}

    converted_dict = np_dict_to_pt(test_dict)

    assert converted_dict["a1"] == np.zeros(4)
    assert converted_dict["a2"] == np.zeros((3, 4))
