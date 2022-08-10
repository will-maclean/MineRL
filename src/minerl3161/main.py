import gym
import minerl
import logging
import torch

print("cuda available: ", torch.cuda.is_available())
logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLObtainDiamondShovel-v0')
obs = env.reset()

counter = 0

while counter < 100:
    action = env.action_space.sample()
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    env.render()

    print("took a random action. step: ", counter)
    counter += 1