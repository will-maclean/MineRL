import pytest

import gym
import minerl


@pytest.fixture(scope="session")
def minerl_env():
    env_name = "MineRLObtainDiamond-v0"
    env = gym.make(env_name)

    env.reset()

    yield env

    env.close()