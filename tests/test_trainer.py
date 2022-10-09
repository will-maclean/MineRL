import dataclasses
import torch
import wandb
import gym
import minerl
from minerl3161.agent import DQNAgent

from minerl3161.hyperparameters import DQNHyperparameters, RainbowDQNHyperparameters
from minerl3161.wrappers import minerlWrapper
from minerl3161.trainer import DQNTrainer, RainbowDQNTrainer


def test_DQNtrainer(minerl_env):
    # just runs main.py for a few steps basically
    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure policy hyperparameters
    hp = DQNHyperparameters()
    # base
    hp.train_steps = 10
    hp.burn_in = 2
    hp.evaluate_every = 11  # will evaluate in the first timestep only
    hp.batch_size = 2
    hp.buffer_size_dataset = 5
    hp.buffer_size_gathered = 5
    hp.checkpoint_every = 11
    hp.feature_names = ["inventory", "pov"]

    # dqn
    hp.model_hidden_layer_size = 6
    hp.mlp_output_size = 6

    # Configure environment
    env = minerlWrapper(minerl_env, **dataclasses.asdict(hp))  #FIXME: surely we need to pass in more shit than this

    # Initialising ActionWrapper to determine number of actions in use
    n_actions = env.action_space.n

    # Configure agent
    agent = DQNAgent(obs_space=env.observation_space, 
       n_actions=n_actions, 
       device=device, 
       hyperparams=hp
       )

    # Initialise trainer and start training
    trainer = DQNTrainer(env=env, agent=agent, hyperparameters=hp, use_wandb=False, device=device)

    print("starting training")
    # run the trainer
    trainer.train()
    print("ending training")

def test_rainbow_trainer(minerl_env):
    # just runs main.py for a few steps basically
    # Loading onto appropriate device
    using_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if using_gpu else "cpu")
    print(f"Loading onto {torch.cuda.get_device_name() if using_gpu else 'cpu'}")

    # Configure policy hyperparameters
    hp = RainbowDQNHyperparameters()
    # base
    hp.train_steps = 10
    hp.burn_in = 2
    hp.evaluate_every = 11  # will evaluate in the first timestep only
    hp.batch_size = 2
    hp.buffer_size_dataset = 5
    hp.buffer_size_gathered = 5
    hp.checkpoint_every = 11
    hp.feature_names = ["inventory", "pov"]

    # dqn
    hp.model_hidden_layer_size = 6
    hp.mlp_output_size = 6

    # Configure environment
    env = minerlWrapper(minerl_env, **dataclasses.asdict(hp))  #FIXME: surely we need to pass in more shit than this

    # Initialising ActionWrapper to determine number of actions in use
    n_actions = env.action_space.n

    # Configure agent
    agent = DQNAgent(obs_space=env.observation_space, 
       n_actions=n_actions, 
       device=device, 
       hyperparams=hp
       )

    # Initialise trainer and start training
    trainer = RainbowDQNTrainer(env=env, agent=agent, hyperparameters=hp, use_wandb=False, device=device)

    print("starting training")
    # run the trainer
    trainer.train()
    print("ending training")
