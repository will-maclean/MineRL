from math import gamma
import sys
import logging
import logging.handlers
import argparse
import os
import torch
import wandb
import gym
from collections import namedtuple

from .agent import DQNAgent
from .trainer import DQNTrainer
from .hyperparameters import DQNHyperparameters

Policy = namedtuple('Policy', ['agent', 'trainer', 'params'])

POLICIES = {
    "dqn": Policy(DQNAgent, DQNTrainer, DQNHyperparameters)
}

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--policy', type=str, default='dqn')
    parser.add_argument('--env', type=str, default="MineRLObtainDiamondVectorObf-v0")
    # parser.add_argument('--weights', type=str)
    # parser.add_argument('--seed', type=int, default=456)
    # parser.add_argument('--resume', default=False, action='store_true')
    # parser.add_argument('--gpu', default=True, action='store_true')
    # parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    # configure paths
    rl_log_dir = os.path.join(args.output_dir, 'rl_logs')
    log_file = os.path.join(rl_log_dir, 'output.log')

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.handlers.RotatingFileHandler(log_file, mode=mode, maxBytes=1000000, backupCount=5)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # configure environment
    env = gym.make(args.env)

    # configure agent
    agent = POLICIES[args.policy].agent

    # configure policy hyperparameters
    hp = POLICIES[args.policy].params

    logging.info(f"Started training of {args.policy} policy in environment {args.env} on device {device}")

    # wandb.init(
    #     project="hri-collab", 
    #     entity="monash-deep-neuron",
    #     config={
    #         "learning_rate": learning_rate,
    #         "batch_size": batch_size,
    #         "buffer_size": buffer_size,
    #         "learning_steps": learning_steps
    #         }
    # )

    # initialise trainer and start training
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, hyperparameters=hp)
    trainer.train()

   
if __name__ == '__main__':
    main()