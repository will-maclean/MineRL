import numpy as np
import pytorch_lightning as pl
import torch as th
from torch.utils.data.dataloader import DataLoader

from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.pl_pretraining import pl_dataset, pl_model
from minerl3161.buffers import ReplayBuffer
from minerl3161.utils import sample_pt_state

def test_pl_dataset():
    # MineRLDataset's basically just wrap a ReplayBuffer
    n_buffer = 1000

    obs_space = {
        "state": np.zeros(5)
    }
    
    buffer = ReplayBuffer(n_buffer, obs_space=obs_space)

    for _ in range(n_buffer):
        buffer.add(
            {"state": np.random.rand(5)},
            np.random.randint(10),
            {"state": np.random.rand(5)},
            np.random.rand(),
            np.rint(np.random.rand())
        )
    
    dataset = pl_dataset.MineRLDataset(buffer)

    assert len(buffer) == len(dataset)

    sample_buffer = buffer[0]
    sample_dataset = dataset[0]

    assert sample_buffer[1] == sample_dataset[1]  # action
    assert sample_buffer[3] == sample_dataset[3]  # reward
    assert sample_buffer[4] == sample_dataset[4]  # done

    # assert states and next states have the same keys in both dataset and buffer
    assert list(sample_buffer[0].keys()) == list(sample_dataset[0].keys()) == \
        list(sample_buffer[2].keys()) == list(sample_dataset[2].keys())  
    
    for key in sample_buffer[0].keys():
        # assert states in both match
        assert (sample_buffer[0][key] == sample_dataset[0][key]).all()

        # assert next states in both match
        assert (sample_buffer[2][key] == sample_dataset[2][key]).all()


def test_pl_model():
    # test we can create the model
    obs_space = {
        "state": np.zeros(5)
    }

    n_actions = 2

    hyperparams = DQNHyperparameters()
    hyperparams.feature_names = list(obs_space.keys())

    model = pl_model.DQNPretrainer(
        obs_space=obs_space,
        n_actions=n_actions,
        hyperparams=hyperparams
    )

    # test we can do a forward pass on the model

    random_state = sample_pt_state(observation_space=obs_space, features=list(obs_space.keys()), batch=3)

    output = model.forward(random_state)

    # create a small dataset 
    n_buffer = 20

    obs_space = {
        "state": np.zeros(5)
    }
    
    buffer = ReplayBuffer(n_buffer, obs_space=obs_space)

    for _ in range(n_buffer):
        buffer.add(
            {"state": np.random.rand(5)},
            np.random.randint(n_actions, size=(1,1)),
            {"state": np.random.rand(5)},
            np.random.rand(),
            np.rint(np.random.rand())
        )
    
    dataset = pl_dataset.MineRLDataset(buffer)

    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_set, val_set = th.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=5, shuffle=False)

    # training
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_loader, val_loader)
