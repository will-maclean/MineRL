import argparse
import dataclasses
import torch
import gym
import minerl
from tqdm import tqdm
from numpy import std
import csv
from random import random, randint

from minerl3161.agent import DQNAgent
from minerl3161.wrappers import minerlWrapper, MineRLWrapper


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env', type=str, default="MineRLNavigateDense-v0")

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='sets if we use gpu hardware')
    
    parser.add_argument('--no-gpu', action='store_false', dest="gpu",
                        help='sets if we use gpu hardware')

    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='number of episodes to evaluate the agent')   

    parser.add_argument('--weights_path', type=str,
                    help='path to trained model weights - must be specified')

    parser.add_argument('--ep_rew_pass', type=str, default=0,
                    help='the rew the agent must obtain in order for the episode to be considered a pass')

    parser.add_argument('--csv_file_path', type=str, default="data/eval_data.csv",
                    help='where the csv file containing the eval data is output')

    parser.add_argument('--mp4_file_path', type=str, default="data/eval_videos",
                    help="where the mp4 file of the agent's eval is output")                       

    args = parser.parse_args()

    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    agent = DQNAgent.load(args.weights_path)
    agent.q1.to(device)
    agent.q1.eval()

    # Configure environment
    env = gym.make(args.env)
    env = minerlWrapper(env, **dataclasses.asdict(agent.hp))

    # TODO: Wrap env in mp4 wrapper
    env = gym.wrappers.Monitor(env, force=True, directory=args.mp4_file_path, video_callable=lambda x: True)
    
    episode_rews = []
    episodes_t_steps = []
    pass_amount = 0
    
    for _ in tqdm(range(args.eval_episodes)):

        t, ep_rew = 0, 0
        state, done = env.reset(), False

        while not done:

            if random() < 0.9:
                action, _ = agent.act(state=state, step=t, train=False)
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)

            ep_rew += reward

            if done:
                episode_rews.append(ep_rew)
                episodes_t_steps.append(t)

                if ep_rew > args.ep_rew_pass: pass_amount += 1

            else:
                state = next_state
                t += 1

    avg_ep_rew = sum(episode_rews)/len(episode_rews)
    avg_ep_t = sum(episodes_t_steps)/len(episodes_t_steps)
    pass_fail_rate = pass_amount/args.eval_episodes  
    rew_std = std(episode_rews)

    headers = ["Average Episode Reward", "Average Episode Length", f"Episode Success Rate (> {args.ep_rew_pass})", "STD of Episode Rewards"]
    data = [avg_ep_rew, avg_ep_t, pass_fail_rate, rew_std]

    with open(args.csv_file_path, 'w') as f: 
        write = csv.writer(f) 
        write.writerow(headers) 
        write.writerow(data) 


if __name__ == '__main__':
    main()