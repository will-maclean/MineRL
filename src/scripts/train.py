import argparse
import dataclasses
from webbrowser import get
from minerl3161.buffer import ReplayBuffer, PrioritisedReplayBuffer
import torch
import wandb
import gym
import minerl
from collections import namedtuple

from minerl3161.agent import DQNAgent, TinyDQNAgent
from minerl3161.trainer import DQNTrainer, RainbowDQNTrainer
from minerl3161.hyperparameters import DQNHyperparameters, RainbowDQNHyperparameters, CartpoleDQNHyperparameters
from minerl3161.wrappers import minerlWrapper, cartPoleWrapper
from minerl3161.termination import get_termination_condition
from minerl3161.hyperparameters import DQNHyperparameters, RainbowDQNHyperparameters
from minerl3161.wrappers import minerlWrapper


Policy = namedtuple('Policy', ['agent', 'trainer', 'wrapper', 'params'])


POLICIES = {
    "vanilla-dqn": Policy(DQNAgent, DQNTrainer, minerlWrapper, DQNHyperparameters),
    "rainbow-dqn": Policy(DQNAgent, RainbowDQNTrainer, minerlWrapper, RainbowDQNHyperparameters),
    "tiny-dqn": Policy(TinyDQNAgent, DQNTrainer, minerlWrapper, DQNHyperparameters),
    "cartpole-dqn": Policy(TinyDQNAgent, DQNTrainer, cartPoleWrapper, CartpoleDQNHyperparameters),
}

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='vanilla-dqn')
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
    
    parser.add_argument('--render', action='store_true', default=False,
                        help='sets if we use gpu hardware')

    args = parser.parse_args()

    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure policy hyperparameters
    hp = POLICIES[args.policy].params()

    # Configure environment
    env = gym.make(args.env)
    env = POLICIES[args.policy].wrapper(env, **dataclasses.asdict(hp))

    # handle human experience
    if args.human_exp_path is None:
        print("WARNING: not using any human experience")
        human_dataset = None
    else:
        human_dataset = PrioritisedReplayBuffer.load(args.human_exp_path) if args.human_exp_path is not None else None

    # Setup termination conditions for the environment (if available)
    termination_conditions = get_termination_condition(args.env)

    # Configure agent
    agent = POLICIES[args.policy].agent(
            obs_space=env.observation_space, 
            n_actions=env.action_space.n, 
            device=device, 
            hyperparams=hp,
            load_path=args.load_path
        )

    if args.wandb:
        wandb.init(
            project=args.env + "-" + args.policy, 
            entity="minerl3161",
            config=hp,
            tags=[args.policy, args.env]
        )

        agent.watch_wandb()

    # Initialise trainer and start training
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, human_dataset=human_dataset, hyperparameters=hp, use_wandb=args.wandb, device=device, render=args.render, termination_conditions=termination_conditions)
    trainer.train()


if __name__ == '__main__':
    main()