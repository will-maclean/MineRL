import argparse
import dataclasses

import gym
import minerl
import numpy as np

from minerl3161.wrappers import MineRLWrapper
from minerl3161.buffer import ReplayBuffer
from minerl3161.hyperparameters import DQNHyperparameters

def convert_dataset(env_name, out_path, hyperparams):

    env = gym.make(env_name)

    _, observation_space = MineRLWrapper.convert_state(observation_space=env.observation_space, **dataclasses.asdict(hyperparams))

    buffer = ReplayBuffer(hyperparams.buffer_size_dataset, observation_space)
    data = minerl.data.make(env_name)
    trajectory_names = data.get_trajectory_names()

    action_set = MineRLWrapper.create_action_set(functional_acts=True, extracted_acts=True)

    for traj_name in trajectory_names:
        state_buffer = np.zeros()
        next_state_buffer = np.zeros()
        for s, a, r, s_, d in data.load_data(traj_name):

            s, _, state_buffer = MineRLWrapper.convert_state(s, state_buffer=state_buffer, **dataclasses.asdict(hyperparams))
            a, _ = MineRLWrapper.map_action(a, action_set=action_set)
            s_, _, next_state_buffer = MineRLWrapper.convert_state(s_, state_buffer=next_state_buffer, **dataclasses.asdict(hyperparams))

            buffer.add(s, a, s_, r, d)

    buffer.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='MineRLObtainDiamond-v0')
    parser.add_argument('--out_path', type=str, default='data/human-xp.pkl')

    args = parser.parse_args()

    hyperparams = DQNHyperparameters()

    convert_dataset(env=args.env_name, out_path=args.out_path, hyperparams=hyperparams)