import argparse
import dataclasses
import os
import gym
import minerl

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import minerl3161
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.buffers import ReplayBuffer
from minerl3161.utils.wrappers import MineRLWrapper
from minerl3161.pl_pretraining.pl_model import DQNPretrainer
from minerl3161.pl_pretraining.pl_dataset import MineRLDataset

def opt():
    
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument("--data_path", type=str, default="human-xp-navigate-dense-PER.pkl")

    # Optional
    parser.add_argument("--env_name", type=str, default="MineRLNavigateDense-v0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target_update_freq", type=int, default=5)

    args = parser.parse_args()

    return args


def main():
    args = opt()

    env = gym.make(args.env_name)
    hp = DQNHyperparameters()
    env = MineRLWrapper(env, **dataclasses.asdict(hp))

    # data
    data = MineRLDataset(ReplayBuffer.load(os.path.join(minerl3161.data_path, args.data_path)))
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
    logger = WandbLogger(project=f"pretraining-{args.env_name}", log_model="all")

    # log gradients, parameter histogram and model topology
    logger.watch(model, log="all")

    # training

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], devices=1, accelerator="gpu", logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
