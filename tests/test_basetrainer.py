from abc import ABC
from minerl3161.agent import BaseAgent, DQNAgent
from minerl3161.hyperparameters import BaseHyperparameters, DQNHyperparameters
from minerl3161.trainer import BaseTrainer
import numpy as np
import os

def test_get_dataset_batches(wrapped_minerl_env):
    
    # os.environ['MINERL_DATA_ROOT'] = '../../../data/human-xp'
    state_space_shape = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }
    n_actions = 32
    hyperparams = DQNHyperparameters()
    hyperparams.feature_names = list(state_space_shape.keys())
    device = "cpu"

    agent = DQNAgent(
        obs_space=state_space_shape, 
        n_actions=n_actions, 
        device=device, 
        hyperparams=hyperparams)
    
    base_trainer = BaseTrainer(wrapped_minerl_env, agent, BaseHyperparameters(), False)
    batches =base_trainer._get_dataset_batches(batch_size=10, num_batches=5)

    assert len(batches) == 5
    assert len(batches[0]['state']['pov']) == 10

def test_sampling(wrapped_minerl_env):

    # os.environ['MINERL_DATA_ROOT'] = '../../../data/human-xp'
    state_space_shape = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }
    n_actions = 32
    hyperparams = DQNHyperparameters()
    hyperparams.feature_names = list(state_space_shape.keys())
    device = "cpu"

    agent = DQNAgent(
        obs_space=state_space_shape, 
        n_actions=n_actions, 
        device=device, 
        hyperparams=hyperparams)

    base_hyper_params = BaseHyperparameters()
    base_trainer = BaseTrainer(wrapped_minerl_env, agent, base_hyper_params, False)

    def strategy(dataset_size, gathered_size, step):
        return dataset_size-step, gathered_size+step

    sample = base_trainer.sample(strategy)

    assert len(sample['reward']) == base_hyper_params.batch_size
    for key in ['reward', 'done', 'action', 'state', 'next_state']:
        assert key in sample
