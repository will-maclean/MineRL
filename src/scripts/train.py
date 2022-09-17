import argparse
from minerl3161.buffer import ReplayBuffer
import torch
import wandb
import gym
import minerl
from collections import namedtuple

from minerl3161.agent import DQNAgent
from minerl3161.trainer import DQNTrainer
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.wrappers import minerlWrapper
from minerl3161.wrappers import MineRLWrapper


Policy = namedtuple('Policy', ['agent', 'trainer', 'params'])


POLICIES = {
    "vanilla-dqn": Policy(DQNAgent, DQNTrainer, DQNHyperparameters)
}

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='vanilla-dqn')
    parser.add_argument('--env', type=str, default="MineRLObtainDiamond-v0")

    # Why can't argparse read bools from the command line? Who knows. Workaround:
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='sets if we use wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest="wandb",
                        help='sets if we use wandb logging')

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='sets if we use gpu hardware')
    parser.add_argument('--no-gpu', action='store_false', dest="gpu",
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
    env = minerlWrapper(env, hp.inventory_feature_names)  #FIXME: surely we need to pass in more shit than this

    load_human_xp(env)

    # Initialising ActionWrapper to determine number of actions in use
    n_actions = env.action_space.n

    # Configure agent
    agent = POLICIES[args.policy].agent(obs_space=env.observation_space, 
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
    trainer = POLICIES[args.policy].trainer(env=env, agent=agent, hyperparameters=hp, use_wandb=args.wandb, device=device)
    trainer.train()


def load_human_xp(env):
    buffer = ReplayBuffer(3000, env.observation_space)
    data = minerl.data.make('MineRLObtainDiamond-v0')
    trajectory_names = data.get_trajectory_names()

    action_set = MineRLWrapper.create_action_set(functional_acts=True, extracted_acts=True)

    for traj_name in trajectory_names:
        for current_state, action, reward, next_state, done in data.load_data(traj_name):
            buffer.add(
                    reward, 
                    done, 
                    MineRLWrapper.map_action(action, action_set), 
                    MineRLWrapper.convert_state(current_state, features=['all']), 
                    MineRLWrapper.convert_state(next_state, features=['all'])
            )

    buffer.save('/opt/project/data/human-xp.pkl')


if __name__ == '__main__':
    main()