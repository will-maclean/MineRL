import gym
import minerl
from minerl3161.hyperparameters import DQNHyperparameters

from minerl3161.wrappers import mineRLObservationSpaceWrapper


def test_env_wrapper_basic_specific(minerl_env):

    w = 16
    h = 16
    features = ['pov', 'iron_pickaxe', 'planks', 'wooden_axe']
    stack = 4

    env = mineRLObservationSpaceWrapper(minerl_env, frame=stack, features=features, downsize_width=w, downsize_height=h)

    assert env.observation_space['pov'].shape == (stack, w, h)

    state = env.reset()

    assert state['pov'].shape == (stack, w, h)
    assert state['inventory'].shape == (3,)

def test_env_wrapper_all(minerl_env):

    n_features = len(list(minerl_env.observation_space['inventory'].spaces.keys()))

    w = 16
    h = 16
    features = DQNHyperparameters().inventory_feature_names
    stack = 4
    env = mineRLObservationSpaceWrapper(minerl_env, frame=stack, features=features, downsize_width=w, downsize_height=h)

    assert env.observation_space['pov'].shape == (stack, w, h)

    state = env.reset()

    assert state['pov'].shape == (stack, w, h)
    assert state['inventory'].shape == (n_features,)
