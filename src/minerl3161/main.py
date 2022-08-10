import argparse
import torch
import wandb
import gym
import minerl
from collections import namedtuple

from minerl3161.agent import DQNAgent
from minerl3161.trainer import DQNTrainer
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.wrappers import MineRLDiscreteActionWrapper


Policy = namedtuple('Policy', ['agent', 'trainer', 'params'])


POLICIES = {
    "vanilla-dqn": Policy(DQNAgent, DQNTrainer, DQNHyperparameters)
}

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='vanilla-dqn')
    parser.add_argument('--env', type=str, default="MineRLObtainDiamondShovel-v0")
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=False)
    args = parser.parse_args()

    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure environment
    env = gym.make(args.env)

    # Configure policy hyperparameters
    hp = POLICIES[args.policy].params()

    # Initialising ActionWrapper to determine number of actions in use
    action_wrapper = MineRLDiscreteActionWrapper(env)
    n_actions = action_wrapper.get_actions_count()

    # Configure agent
    agent = POLICIES[args.policy].agent(env.observation_space, n_actions, device, hp)

    if args.wandb:
        pass
        # TODO: Setup wandb project
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

    # Initialise trainer and start training
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, hyperparameters=hp)
    trainer.train()

   
if __name__ == '__main__':
    main()