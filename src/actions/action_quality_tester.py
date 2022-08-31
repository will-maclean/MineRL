import gym
import minerl
import os
import numpy as np
import pickle
from tqdm import tqdm

import minerl3161


ITERS = 100000
NUM_ACTIONS = 12


def get_actions(obf: bool):
    file_name = f'actions-Treechop.{"npy" if obf else "pickle"}'
    filepath = os.path.join(minerl3161.actions_path, file_name)
    
    if obf:
        action_set = np.load(filepath)
        actions = [{"vector": action} for action in action_set]
    else:
        with open(filepath, 'rb') as f:
            actions = pickle.load(f)

    return actions


def test_actions(obf: bool):
    env = gym.make(f'MineRLTreechop{"VectorObf" if obf else ""}-v0')
    env.reset()
    total_reward = 0

    actions = get_actions(obf)

    for _ in tqdm(range(ITERS)):
        action = actions[np.random.choice(NUM_ACTIONS)]
        _, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            env.reset()
    env.close()
    
    print(f"Rew after {ITERS} time steps: {total_reward} on {'obf' if obf else 'non obf'} actions.")


test_actions(obf=True)
test_actions(obf=False)

