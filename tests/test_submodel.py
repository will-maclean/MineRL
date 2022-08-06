import torch as th
import numpy as np
import pytest

from minerl3161.submodel import CNN, MLP, MineRLFeatureExtraction
from minerl3161.utils import sample_pt_state


def test_cnn():
    input_shape = (3, 64, 64)

    cnn = CNN(input_shape)

    sample_input = th.randn((1, *input_shape))

    sample_output = cnn(sample_input)


def test_mlp():
    input_shape = 64
    output_size = 10

    layers = [(), (64,), (64, 64)]
    for layer in layers:
        mlp = MLP(input_shape, output_size, layer)
        sample_input = th.randn((1, input_shape))
        sample_output = mlp(sample_input)
        assert sample_output.shape[1] == output_size


def test_minerl_feature_extraction_subnet():
    sample_observation_space = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }

    feature_names = ["pov", "f2"]

    sample = sample_pt_state(sample_observation_space, feature_names, batch=1)

    feature_extraction = MineRLFeatureExtraction(
        sample_observation_space, feature_names
    )

    sample_out = feature_extraction(sample)

    assert len(sample_out.shape) == 2
    assert sample_out.shape[0] == 1