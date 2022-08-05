import gym
import minerl

from minerl3161.wrappers import mineRLObservationSpaceWrapper

def test_env_wrapper():
    env_name = "MineRLObtainDiamondShovel-v0"
    unwrapped_env = gym.make(env_name)

    w = 16
    h = 16
    camera_feature = 'pov'
    stack = 4

    env = mineRLObservationSpaceWrapper(unwrapped_env, frame=stack, camera_feature_name=camera_feature, downsize_width=w, downsize_height=h)

    assert env.observation_space[camera_feature].shape == (stack, w, h)

    state = env.reset()

    assert state[camera_feature].shape == (stack, w, h)