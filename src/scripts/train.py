import argparse
import dataclasses
from minerl3161.buffer import ReplayBuffer, PrioritisedReplayBuffer
import torch
import wandb
import gym
import minerl
from collections import namedtuple

from minerl3161.agent import DQNAgent, TinyDQNAgent
from minerl3161.trainer import DQNTrainer, RainbowDQNTrainer
from minerl3161.hyperparameters import DQNHyperparameters, RainbowDQNHyperparameters
from minerl3161.wrappers import minerlWrapper
from minerl3161.wrappers import MineRLWrapper
from os.path import exists

Policy = namedtuple('Policy', ['agent', 'trainer', 'params'])


POLICIES = {
    "vanilla-dqn": Policy(DQNAgent, DQNTrainer, DQNHyperparameters),
    "rainbow-dqn": Policy(DQNAgent, RainbowDQNTrainer, RainbowDQNHyperparameters),
    "tiny-dqn": Policy(TinyDQNAgent, DQNTrainer, DQNHyperparameters)
}

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='vanilla-dqn')
    parser.add_argument('--env', type=str, default="MineRLObtainDiamondDense-v0")

    # Why can't argparse read bools from the command line? Who knows. Workaround:
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='sets if we use wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest="wandb",
                        help='sets if we use wandb logging')

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='sets if we use gpu hardware')
    
    parser.add_argument('--no-gpu', action='store_false', dest="gpu",
                        help='sets if we use gpu hardware')

    parser.add_argument('--human_exp_path', type=str, default="data/human-xp.pkl",
                        help='pass in path to human experience pickle')

    args = parser.parse_args()

    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure policy hyperparameters
    hp = POLICIES[args.policy].params()

    # Configure environment
    env = gym.make(args.env)
    env = minerlWrapper(env, **dataclasses.asdict(hp))


    human_dataset = PrioritisedReplayBuffer.load(args.human_exp_path)

    # Initialising ActionWrapper to determine number of actions in use
    n_actions = env.action_space.n

    # Configure agent
    agent = POLICIES[args.policy].agent(
            obs_space=env.observation_space, 
            n_actions=n_actions, 
            device=device, 
            hyperparams=hp
        )

    if args.wandb:
        wandb.init(
            project="diamond-pick", 
            entity="minerl3161",
            config=hp
        )

    # Initialise trainer and start training
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, human_dataset=human_dataset, hyperparameters=hp, use_wandb=args.wandb, device=device)
    trainer.train()


if __name__ == '__main__':
    main()