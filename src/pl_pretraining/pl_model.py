from copy import deepcopy
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F

from minerl3161.models import DQNNet


class DQNPretrainer(pl.LightningModule):
    def __init__(self, obs_space, n_actions, hyperparams, gamma=0.99, target_update_freq=1) -> None:
        super().__init__()

        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.q1 = DQNNet(
            state_shape=obs_space,
            n_actions=n_actions, 
            dqn_hyperparams=hyperparams,
            layer_size=hyperparams.model_hidden_layer_size
            )
        self.q2 = deepcopy(self.q1)
        self.q2.eval()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        s, a, s_, r, d = batch
        loss = self._calc_loss(s, a, s_, r, d)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with th.no_grad():
            s, a, s_, r, d = batch
            loss = self._calc_loss(s, a, s_, r, d)
            self.log("train_loss", loss)
            return loss

    def _calc_loss(self, s, a, s_, r, d):
        # estimate q values for current states/actions using q1 network
        q_values = self.q1(s)
        q_values = q_values.gather(1, a)

        # estimate q values for next states/actions using q2 network
        next_actions = self.q2(s_).argmax(dim=1).unsqueeze(1)
        next_q_values = self.q2(s_).gather(1, next_actions)

        # calculate TD target for Bellman Equation
        td_target = r + self.hp.gamma * next_q_values * (1 - d)

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = F.smooth_l1_loss(q_values, td_target)

        return loss

    def on_train_epoch_end(self, trainer: pl.Trainer) -> None:
        if trainer.current_epoch % self.update_freq == 0:
            self.q2.load_state_dict(self.q1.state_dict())