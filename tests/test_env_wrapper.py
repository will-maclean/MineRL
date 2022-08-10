import gym
import minerl

from minerl3161.wrappers import mineRLObservationSpaceWrapper

def test_env_wrapper():
    env_name = "MineRLObtainDiamondShovel-v0"
    unwrapped_env = gym.make(env_name)

    w = 16
    h = 16
    features = ['pov', 'stone_sword', 'stonecutter', 'stone_shovel']
    stack = 4

    env = mineRLObservationSpaceWrapper(unwrapped_env, frame=stack, features=features, downsize_width=w, downsize_height=h)

    assert env.observation_space['pov'].shape == (stack, w, h)

    state = env.reset()

    assert state['pov'].shape == (stack, w, h)
    assert state['inventory'].shape == (3,)