import pytest

import gym
import minerl
from minerl3161.hyperparameters import DQNHyperparameters

from minerl3161.wrappers import mineRLObservationSpaceWrapper


@pytest.fixture(scope="session")
def minerl_env():
    env_name = "MineRLObtainDiamond-v0"
    env = gym.make(env_name)

    env.reset()

    yield env

    env.close()


@pytest.fixture(scope='session')
def wrapped_minerl_env():
    
    env_name = "MineRLObtainDiamond-v0"
    env = gym.make(env_name)
    
    w = 16
    h = 16
    stack = 4

    hyperparams = DQNHyperparameters()
    env = mineRLObservationSpaceWrapper(env, frame=stack, features=hyperparams.inventory_feature_names, downsize_width=w, downsize_height=h)

    env.reset()

    yield env

    env.close()
