import gym
import minerl

from minerl3161.wrappers import mineRLObservationSpaceWrapper

def test_env_wrapper():
    env_name = "MineRLObtainDiamondShovel-v0"
    env = gym.make(env_name)
    env = mineRLObservationSpaceWrapper(env)

    state = env.reset()

    print(env.observation_space.shape)
    print(state)