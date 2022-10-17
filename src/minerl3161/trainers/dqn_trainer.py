from typing import Union, List
from time import perf_counter

import gym
import torch as th
from torch.optim import Adam

from minerl3161.agents import BaseAgent
from minerl3161.hyperparameters import DQNHyperparameters
from minerl3161.buffers import ReplayBuffer
from minerl3161.utils.termination import TerminationCondition
from minerl3161.utils.utils import np_batch_to_tensor_batch, copy_weights

from .base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """
    The Trainer for the DQNAgent. Inherits from the BaseTrainer class, implementing the _train_step() method.
    """

    def __init__(
        self, 
        env: gym.Env, 
        agent: BaseAgent, #TODO: should this be a DQNAgent?
        hyperparameters: DQNHyperparameters, 
        human_dataset: Union[ReplayBuffer, None] = None, 
        use_wandb: bool = False, 
        device: str = "cpu", 
        render: bool = False, 
        termination_conditions: Union[List[TerminationCondition], None] = None,
        capture_eval_video: bool = True
    ) -> None:
        """
        Initialiser for DQNTrainer

        Args:
            env (gym.Env): environmnet to train in
            agent (BaseAgent): agent to train
            hyperparameters (DQNHyperparameters): hyperparameters to train with
            human_dataset (Union[ReplayBuffer, None]): a ReplayBuffer containing expert human transitions
            use_wandb (bool): dictates whether wandb should be used or not
            device (str): dictates what device the tensors should be loaded onto for training the model 
            render (bool): dictates whether the envrionment should be rendered during training or not
            termination_conditions (Union[List[TerminationCondition], None]): the conditions that dictate when training should conclude
            capture_eval_video (bool): dictates whether a video should be captured when performing the eval callback
        """
        super().__init__(
            env=env, 
            agent=agent, 
            human_dataset=human_dataset, 
            hyperparameters=hyperparameters, 
            use_wandb=use_wandb, device=device, 
            render=render, 
            termination_conditions=termination_conditions, 
            capture_eval_video=capture_eval_video
        )

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=self.hp.lr)

    def _train_step(self, step: int) -> None:
        """
        Implements the network training that is to be completed at each train step

        Args:
            step (int): the current time step in the training
        """
        log_dict = {}
        start_time = perf_counter()

        # Get a batch of experience from the gathered transitions
        batch = self.sample(self.hp.sampling_strategy)

        batch = np_batch_to_tensor_batch(batch, self.device)

        # calculate loss signal
        loss = self._calc_loss(batch)

        # update model parameters
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # do q2 model update (also referred to as target network)
        if (
            self.hp.hard_update_freq != 0
            and step % self.hp.hard_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=None)
        if (
            self.hp.soft_update_freq != 0
            and step % self.hp.soft_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=self.hp.polyak_tau)
        
        end_time = perf_counter()
        log_dict["train_fps"] = 1 / (end_time - start_time)

        return {"loss": loss.detach().cpu().item()}

    def _calc_loss(self, batch: dict) -> th.Tensor:
        """
        Used to calculate the loss of the supplied batch (the margin of error between the q-values that model currently predicts, and
        the newly calculated ones)

        Args:
            batch (dict): the batch of data extracted from the ReplayBuffer, being used to train the data
        
        Returns:
            th.Tensor: the average loss of each of the samples in the batch
        """
        # estimate q values for current states/actions using q1 network
        q_values = self.agent.q1(batch["state"])
        q_values = q_values.gather(1, batch["action"])

        # estimate q values for next states/actions using q2 network
        next_q_values = self.agent.q2(batch["next_state"])
        next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_q_values = next_q_values.gather(1, next_actions)

        # calculate TD target for Bellman Equation
        td_target = self.hp.reward_scale * th.sign(batch[
            "reward"
        ]) + self.hp.gamma * next_q_values * (1 - batch["done"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = th.nn.functional.smooth_l1_loss(q_values, td_target)

        return loss
    