""" BaseTrainer and implementations stored here
"""
from abc import abstractmethod

import gym
import numpy as np
import torch
from minerl3161.utils import copy_weights
from torch.optim import Adam, Optimizer

from .agent import BaseAgent
from .buffer import ReplayBuffer
from .evaluator import Evaluator
from .hyperparameters import BaseHyperparameters, DQNHyperparameters


# TODO: write tests
class BaseTrainer:
    """Abstract class for Trainers. At the least, all implementations must have _train_step()."""

    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: BaseHyperparameters
    ) -> None:
        """Initialiser for BaseTrainer.

        Args:
            env (gym.Env): environmnet to train in
            agent (BaseAgent): agent to train
            hyperparameters (BaseHyperparameters): hyperparameters to train with
        """
        self.env = env
        self.agent = agent
        self.hp = hyperparameters

        self.gathered_transitions = ReplayBuffer(self.hp.buffer_size_gathered)
        self.dataset_transitions = ReplayBuffer(self.hp.buffer_size_dataset)
        self.evaluator = Evaluator(env)

    def train(self) -> None:
        """main training function. This basic training loop should be enough for most conventional RL algorithms"""

        t = 0
        while t < self.hp.train_steps:
            log_dict = {"step": t}

            if t % self.hp.gather_every == 0:
                log_dict += self._gather(self.hp.gather_n)

            if t > self.hp.burn_in and t % self.hp.train_every == 0:
                log_dict += self._train_step(t)

            if t % self.hp.evaluate_every == 0:
                log_dict += self.evaluator.evaluate(
                    self.agent, self.hp.evaluate_episodes
                )

            log_dict += self._housekeeping(t)

            self._log(log_dict)

    def _gather(self, steps: int) -> None:
        """gathers steps of experience from the environment

        Args:
            steps (int): how many steps of experience to gather
        """
        pass

    @abstractmethod
    def _train_step(self, step: int) -> None:
        raise NotImplementedError()

    def _housekeeping(self, step: int) -> None:
        # TODO: Better name for this method
        pass

    def _log(self, log_dict: dict) -> None:
        pass


# TODO: write tests
class DQNTrainer(BaseTrainer):
    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: DQNHyperparameters
    ) -> None:
        super().__init__(env=env, agent=agent, hyperparameters=hyperparameters)

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=self.hp.lr)

    def _train_step(self, step: int) -> None:
        # Get a batch of experience from the gathered transitions
        batch = self.gathered_transitions.sample(
            self.batch_size
        )  # TODO: implement strategy to sample from both buffers

        # convert np transitions into torch tensors
        batch["states"] = torch.from_numpy(batch["states"]).to(
            self.device, dtype=torch.float32
        )
        batch["next_states"] = torch.from_numpy(batch["next_states"]).to(
            self.device, dtype=torch.float32
        )
        batch["actions"] = torch.tensor(
            batch["actions"], dtype=torch.long, device=self.device
        )
        batch["rewards"] = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        batch["dones"] = torch.tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )

        # calculate loss signal
        loss = self._calc_loss(batch)

        # update model parameters
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # do q2 model update (also referred to as target network)
        if (
            self.hp.hard_update_freq is not None
            and step % self.hp.hard_update_freq == 0
        ):
            copy_weights(self.net, self.target_net, polyak=None)
        if (
            self.hp.soft_update_freq is not None
            and step % self.hp.soft_update_freq == 0
        ):
            copy_weights(self.q1, self.q2, polyak=self.hp.polyak_tau)

        return {"loss": loss.detach().cpu().item()}

    def _calc_loss(self, batch: dict) -> torch.Tensor:
        # estimate q values for current states/actions using q1 network
        q_values = self.q1(batch["states"])
        actions = actions.unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        # estimate q values for next states/actions using q2 network
        next_actions = self.q2(batch["next_states"]).argmax(dim=1).unsqueeze(1)
        next_q_values = self.q2(batch["next_states"]).gather(1, next_actions).squeeze(1)

        # calculate TD target for Bellman Equation
        td_target = self.reward_scale * batch[
            "rewards"
        ] + self.gamma * next_q_values * (1 - batch["dones"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target)

        return loss
