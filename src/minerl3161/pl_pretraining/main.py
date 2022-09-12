import argparse
import gym
import minerl
from minerl.data import BufferedBatchIter

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from minerl3161.hyperparameters import DQNHyperparameters

from minerl3161.pl_pretraining.pl_model import DQNPretrainer
from minerl3161.wrappers import mineRLObservationSpaceWrapper


def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="data/MineRLObtainDiamond-v0")

    args = parser.parse_args()

    return args


def main():
    hp = DQNHyperparameters.inventory_feature_names
    env_name = "MineRLObtainDiamond-v0"
    env = gym.make(env_name)
    env = mineRLObservationSpaceWrapper(env, )

    args = opt()

    # data
    data = minerl.data.make('MineRLObtainDiamond-v0')
    iterator = BufferedBatchIter(data)

    # TODO: create dataloader(?)

    train_set, val_set = dataset.get_train_val()
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)

    # model
    model = DQNPretrainer(
        obs_space=env.observation_space,
        n_actions=env.action_space.n,
        hyperparams=hp,
        gamma=args.gamma,
    )

    # checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")

    # training
    trainer = pl.Trainer(epochs=args.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
