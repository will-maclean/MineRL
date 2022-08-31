from abc import ABC
from minerl3161.agent import BaseAgent
from minerl3161.hyperparameters import BaseHyperparameters
from ..src.minerl3161.trainer import BaseTrainer
import os

def test_get_dataset_batches(minerl_env):
    
    # os.environ['MINERL_DATA_ROOT'] = '../../../data/human-xp'

    base_trainer = BaseTrainer(minerl_env, BaseAgent(ABC()), BaseHyperparameters(), False)
    batches =base_trainer._get_dataset_batches(batch_size=10, num_batches=5)

    assert len(batches) == 5
    assert len(batches[0]['state']['pov']) == 10

def test_sampling(minerl_env):

    # os.environ['MINERL_DATA_ROOT'] = '../../../data/human-xp'

    base_hyper_params = BaseHyperparameters()
    base_trainer = BaseTrainer(minerl_env, BaseAgent(ABC()), base_hyper_params, False)

    def strategy(dataset_size, gathered_size, step):
        dataset_size -= step
        gathered_size += step

    sample = base_trainer.sample(strategy)

    assert len(sample) == base_hyper_params.batch_size
    for key in ['reward', 'done', 'action', 'state', 'next_state']:
        assert key in sample
