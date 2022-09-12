import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from minerl3161.pl_pretraining.pl_model import DQNPretrainer


def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    return args


def main():

    args = opt()

    # data
    dataset = PLMinerlData()
    train_set, val_set = dataset.get_train_val()
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)
    # model
    model = DQNPretrainer(
        gamma=args.gamma,
    )
    # training
    trainer = pl.Trainer()
    trainer.fit(model, train_loader, val_loader)
