import gym
import minerl
import os
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import minerl3161


ITERS = 10000
NUM_ACTIONS = 12


def get_actions():
    file_name = f'actions-ObtainDiamond.pickle'
    filepath = os.path.join(minerl3161.actions_path, file_name)

    with open(filepath, 'rb') as f:
            actions = pickle.load(f)

    return actions


def test_actions():
    fig, ax = plt.subplots()

    env = gym.make(f'MineRLNavigateDense-v0')
    env.reset()
    total_reward = 0
    rewards = []

    actions = get_actions()


    for _ in tqdm(range(ITERS)):
        action = actions[np.random.choice(NUM_ACTIONS)]
        _, reward, done, _ = env.step(action)
        env.render()

        total_reward += reward
        rewards.append(reward)

        if done:
            # env.reset()
            break
    # env.close()

    ax.plot(rewards)
    plt.show()
    
    print(f"Rew after {ITERS} time steps: {total_reward}")


# test_actions(obf=True)
test_actions(obf=False)

