import gym
import minerl
from minerl3161.hyperparameters import DQNHyperparameters

from minerl3161.utils.wrappers import minerlWrapper


def test_env_wrapper_default(minerl_env):

        env = minerlWrapper(minerl_env)
    
        assert env.observation_space['pov'].shape == (4, 64, 64)
    
        state = env.reset()
    
        assert state['pov'].shape == (4, 64, 64)
        assert len(state['inventory'].shape) == 1


def test_env_wrapper_basic_specific(minerl_env):

    w = 16
    h = 16
    inventory_feature_names = ['iron_pickaxe', 'planks', 'wooden_axe']
    stack = 4

    env = minerlWrapper(minerl_env, n_stack=stack, inventory_feature_names=inventory_feature_names, resize_w=w, resize_h=h)

    assert env.observation_space['pov'].shape == (stack, w, h)

    state = env.reset()

    assert state['pov'].shape == (stack, w, h)
    assert state['inventory'].shape == (3,)

def test_env_wrapper_all(minerl_env):

    n_features = len(list(minerl_env.observation_space['inventory'].spaces.keys()))

    w = 16
    h = 16
    inventory_feature_names = DQNHyperparameters().inventory_feature_names
    stack = 4
    env = minerlWrapper(minerl_env, n_stack=stack, inventory_feature_names=inventory_feature_names, resize_w=w, resize_h=h)

    assert env.observation_space['pov'].shape == (stack, w, h)

    state = env.reset()

    assert state['pov'].shape == (stack, w, h)
    assert state['inventory'].shape == (n_features,)
