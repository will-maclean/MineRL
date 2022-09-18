import argparse
import dataclasses
import gym

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.buffer import ReplayBuffer
from minerl3161.wrappers import MineRLWrapper

from pl_model import DQNPretrainer
from pl_dataset import MineRLDataset

def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target_update_freq", type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = opt()

    env_name = "MineRLObtainDiamond-v0"
    env = gym.make(env_name)
    hp = DQNHyperparameters()
    env = MineRLWrapper(env, **dataclasses.asdict(hp))

    # data
    data = MineRLDataset(ReplayBuffer.load(args.data_path))
    n_train = int(len(data) * args.train_val_split)
    n_val = len(data) - n_train
    train_set, val_set = th.utils.data.random_split(data, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = DQNPretrainer(
        obs_space=env.observation_space,
        n_actions=env.action_space.n,
        hyperparams=hp,
        gamma=args.gamma,
    )

    # delete the env to save some space
    del env

    # checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    logger = WandbLogger()

    # training

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], devices=1, accelerator="gpu", logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
