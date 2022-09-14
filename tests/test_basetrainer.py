from abc import ABC
from minerl3161.agent import DQNAgent
from minerl3161.hyperparameters import BaseHyperparameters, DQNHyperparameters
from minerl3161.trainer import BaseTrainer, DQNTrainer
import numpy as np
import os

from minerl3161.wrappers import minerlWrapper

def test_get_dataset_batches(minerl_env):

    wrapped_minerl_env = minerlWrapper(minerl_env)
    
    device="cpu"

    hyperparams = DQNHyperparameters()
    hyperparams.checkpoint_every = None  # don't checkpoint, as it will fail without wandb

    agent = DQNAgent(
        obs_space=wrapped_minerl_env.observation_space, 
        n_actions=wrapped_minerl_env.action_space.n, 
        device=device, 
        hyperparams=hyperparams)
    
    base_trainer = DQNTrainer(wrapped_minerl_env, agent, hyperparams, use_wandb=False)
    batches = base_trainer._get_dataset_batches(batch_size=10, num_batches=5)

    assert len(batches) == 5
    assert len(batches[0]['state']['pov']) == 10

def test_sampling(minerl_env):

    wrapped_minerl_env = minerlWrapper(minerl_env)

    hyperparams = DQNHyperparameters()
    hyperparams.checkpoint_every = None  # don't checkpoint, as it will fail without wandb
    device = "cpu"

    agent = DQNAgent(
        obs_space=wrapped_minerl_env.observation_space, 
        n_actions=wrapped_minerl_env.action_space.n, 
        device=device, 
        hyperparams=hyperparams)

    base_trainer = DQNTrainer(wrapped_minerl_env, agent, hyperparams, use_wandb=False)

    def strategy(dataset_size, gathered_size, step):
        return dataset_size-step, gathered_size+step

    sample = base_trainer.sample(strategy)

    assert len(sample['reward']) == hyperparams.batch_size
    for key in ['reward', 'done', 'action', 'state', 'next_state']:
        assert key in sample
