import gym
import minerl
import os
import numpy as np
import pickle
from tqdm import tqdm
# import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT_PATH = Path(__file__).absolute().parent.parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')
ACTIONS_PATH = SRC_PATH.joinpath('actions')

ITERS = 10000
NUM_ACTIONS = 28

def get_actions():
    file_name = f'all-actions.pickle'
    filepath = os.path.join(ACTIONS_PATH, file_name)

    with open(filepath, 'rb') as f:
            actions = pickle.load(f)

    return actions


def test_actions():
    # fig, ax = plt.subplots()

    env = gym.make(f'MineRLNavigateDense-v0')
    env.reset()
    total_reward = 0
    rewards = []

    actions = get_actions()
    print("Retrieved Actions")

    for _ in tqdm(range(ITERS)):
        action = np.random.choice(actions)
        _, reward, done, _ = env.step(action)
        env.render()

        total_reward += reward
        rewards.append(reward)

        if done:
            # env.reset()
            break
    # env.close()

    # ax.plot(rewards)
    # plt.show()
    
    print(f"Rew after {ITERS} time steps: {total_reward}")


# test_actions(obf=True)
test_actions()

