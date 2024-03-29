from copy import deepcopy
from typing import Any, Dict
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F

import gym
from minerl3161.hyperparameters.minerl_dqn_hp import MineRLDQNHyperparameters

from minerl3161.models.DQNNetworks import DQNNet

class DQNPretrainer(pl.LightningModule):
    """Provides DQN pretraining using PyTorch lightning
    """
    def __init__(
        self, 
        obs_space: Dict[str, gym.Space], 
        n_actions: int, 
        hyperparams: MineRLDQNHyperparameters, 
        gamma: float = 0.99, 
        target_update_freq: int = 1, 
        lr: float = 3e-4
    ) -> None:
        """Constructor

        Args:
            obs_space (Dict[str, gym.Space]): observation space to be userd
            n_actions (int): number of actions
            hyperparams (DQNHyperparameters): hyperparam object
            gamma (float, optional): gamma constant. Defaults to 0.99.
            target_update_freq (int, optional): update target network every n epochs. Defaults to 1.
            lr (float, optional): initial learning rate. Defaults to 3e-4.
        """
        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.q1 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions, 
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size
            )
        self.q2 = deepcopy(self.q1)
        self.q2.requires_grad_(False)

    def forward(self, x: Dict[str, th.tensor]) -> th.tensor:
        """forward for model

        Args:
            x (Dict[str, th.tensor]): input data

        Returns:
            th.tensor: output data
        """
        return self.q1(x)

    def training_step(self, batch: Any, batch_idx: int) -> th.tensor:
        """complete a single train step

        Args:
            batch (Any): batch of data
            batch_idx (int): batch idx

        Returns:
            th.tensor: loss tensor
        """
        s, a, s_, r, d = batch
        loss = self._calc_loss(s, a, s_, r, d)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx) -> th.tensor:
        """complete a single validation step

        Args:
            batch (Any): batch of data
            batch_idx (int): batch idx

        Returns:
            th.tensor: loss tensor
        """

        with th.no_grad():
            s, a, s_, r, d = batch
            loss = self._calc_loss(s, a, s_, r, d)
            self.log("val_loss", loss, on_epoch=True, on_step=False)
            return loss

    def _calc_loss(self, s: Dict[str, th.tensor], a: th.tensor, s_: Dict[str, th.tensor], r: th.tensor, d: th.tensor) -> th.tensor:
        """calculate loss for a given sample

        Args:
            s (Dict[str, th.tensor]): state
            a (th.tensor): action
            s_ (Dict[str, th.tensor]): next state
            r (th.tensor): reward
            d (th.tensor): done

        Returns:
            th.tensor: loss
        """
        # type checking
        a = a.to(th.long)
        d = d.to(th.float32)

        # estimate q values for current states/actions using q1 network
        q_values = self.q1(s)
        q_values = q_values.gather(1, a)

        # estimate q values for next states/actions using q2 network

        next_q_values = self.q2(s_)
        next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_q_values = next_q_values.gather(1, next_actions)

        # calculate TD target for Bellman Equation
        td_target = th.sign(r) + self.gamma * next_q_values * (1 - d)

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = th.nn.functional.smooth_l1_loss(q_values, td_target)

        return loss

    def on_train_epoch_end(self) -> None:
        """callback for end of training epoch
        """
        if self.current_epoch % self.target_update_freq == 0:
            self.q2.load_state_dict(self.q1.state_dict())
    
    def configure_optimizers(self):
        """sets up optimisers

        Returns:
            _type_: optimiser to use
        """
        return th.optim.Adam(self.q1.parameters(), lr=self.lr)