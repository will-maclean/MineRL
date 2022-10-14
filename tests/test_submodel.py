import torch as th
import numpy as np
import gym
import minerl

from minerl3161.models.submodel import CNN, MLP, MineRLFeatureExtraction
from minerl3161.utils import sample_pt_state
from minerl3161.wrappers import minerlWrapper



def test_cnn():
    input_shape = (3, 64, 64)

    cnn = CNN(input_shape)

    sample_input = th.randn((1, *input_shape))

    _ = cnn(sample_input)


def test_mlp():
    input_shape = 64
    output_size = 10

    layers = [(), (64,), (64, 64)]
    for layer in layers:
        mlp = MLP(input_shape, output_size, layer)
        sample_input = th.randn((1, input_shape))
        sample_output = mlp(sample_input)
        assert sample_output.shape[1] == output_size


def test_minerl_feature_extraction_subnet_dummy():
    # dummy data
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


def test_minerl_feature_extraction_subnet_real(minerl_env):
    # real observation space
    w = 16
    h = 16
    features = ['stone_sword', 'stonecutter', 'stone_shovel']
    stack = 4

    env = minerlWrapper(minerl_env, n_stack=stack, features=features, resize_w=w, resize_h=h)

    obs_space = env.observation_space


    feature_extraction = MineRLFeatureExtraction(
            obs_space, ['pov', 'inventory']
        )