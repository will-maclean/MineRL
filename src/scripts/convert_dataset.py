import argparse
import dataclasses

from tqdm import tqdm
import gym
import minerl
import numpy as np

from minerl3161.wrappers import MineRLWrapper
from minerl3161.buffer import ReplayBuffer
from minerl3161.hyperparameters import DQNHyperparameters

def convert_dataset(env_name, out_path, hyperparams, save_every=5):

    env = gym.make(env_name)

    _, observation_space, empty_buffer = MineRLWrapper.convert_state(observation_space=env.observation_space, **dataclasses.asdict(hyperparams))

    buffer = ReplayBuffer(hyperparams.buffer_size_dataset, observation_space)
    data = minerl.data.make(env_name)
    trajectory_names = data.get_trajectory_names()

    action_set = MineRLWrapper.create_action_set(functional_acts=True, extracted_acts=True)
    
    counter = 0

    for traj_name in tqdm(trajectory_names):
        state_buffer = np.zeros_like(empty_buffer)
        next_state_buffer = np.zeros_like(empty_buffer)
        for s, a, r, s_, d in data.load_data(traj_name):


            s, _, state_buffer = MineRLWrapper.convert_state(s, state_buffer=state_buffer, **dataclasses.asdict(hyperparams))
            a = MineRLWrapper.map_action(a, action_set=action_set)
            s_, _, next_state_buffer = MineRLWrapper.convert_state(s_, state_buffer=next_state_buffer, **dataclasses.asdict(hyperparams))

            counter += 1
            
            if counter % save_every == 0:
                buffer.add(s, a, s_, r, d)

            if buffer.full:
                break
        
        if buffer.full:
                break

    buffer.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='MineRLObtainDiamond-v0')
    parser.add_argument('--out_path', type=str, default='data/human-xp.pkl')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    hyperparams = DQNHyperparameters()

    convert_dataset(env_name=args.env_name, out_path=args.out_path, hyperparams=hyperparams, save_every=args.save_every)