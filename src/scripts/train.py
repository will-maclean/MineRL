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
    parser.add_argument('--policy', type=str, default='rainbow-dqn')
    parser.add_argument('--env', type=str, default="MineRLNavigateDense-v0")

    # Why can't argparse read bools from the command line? Who knows. Workaround:
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='sets if we use wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest="wandb",
                        help='sets if we use wandb logging')

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='sets if we use gpu hardware')
    
    parser.add_argument('--no-gpu', action='store_false', dest="gpu",
                        help='sets if we use gpu hardware')

    parser.add_argument('--human_exp_path', type=str, default=None,
                        help='pass in path to human experience pickle')
    
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to model checkpoint to load (optional)')

    args = parser.parse_args()

    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure policy hyperparameters
    hp = POLICIES[args.policy].params()

    # Configure environment
    env = gym.make(args.env)
    env = minerlWrapper(env, repeat_action=5, **dataclasses.asdict(hp))

    # handle human experience
    if args.human_exp_path is None:
        print("WARNING: not using any human experience")
        human_dataset = None
    else:
        human_dataset = PrioritisedReplayBuffer.load(args.human_exp_path) if args.human_exp_path is not None else None

    # Initialising ActionWrapper to determine number of actions in use
    n_actions = env.action_space.n

    # Configure agent
    agent = POLICIES[args.policy].agent(
            obs_space=env.observation_space, 
            n_actions=n_actions, 
            device=device, 
            hyperparams=hp,
            load_path=args.load_path
        )

    if args.wandb:
        wandb.init(
            project="diamond-pick", 
            entity="minerl3161",
            config=hp,
            tags=[args.env, args.policy]
        )

    # Initialise trainer and start training
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, human_dataset=human_dataset, hyperparameters=hp, use_wandb=args.wandb, device=device)
    trainer.train()


if __name__ == '__main__':
    main()