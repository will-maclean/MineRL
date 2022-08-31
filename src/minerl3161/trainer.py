""" BaseTrainer and implementations stored here
"""
from abc import abstractmethod
from typing import Any, Dict
import time

import gym
import numpy as np
import wandb
import torch
from minerl3161.utils import copy_weights
from torch.optim import Adam, Optimizer

from .agent import BaseAgent
from .buffer import ReplayBuffer
from .evaluator import Evaluator
from .hyperparameters import BaseHyperparameters, DQNHyperparameters

from minerl.data import BufferedBatchIter

# TODO: write tests
class BaseTrainer:
    """Abstract class for Trainers. At the least, all implementations must have _train_step()."""

    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: BaseHyperparameters, use_wandb: bool =False
    ) -> None:
        """Initialiser for BaseTrainer.

        Args:
            env (gym.Env): environmnet to train in
            agent (BaseAgent): agent to train
            hyperparameters (BaseHyperparameters): hyperparameters to train with
        """
        self.env: gym.Env = env
        self.agent: BaseAgent = agent
        self.hp: BaseHyperparameters = hyperparameters
        self.use_wandb = use_wandb

        self.gathered_transitions = ReplayBuffer(
            self.hp.buffer_size_gathered, self.env.observation_space
        )

        data = minerl.data.make('MineRLObtainDiamond-v0')
        self.dataset_iter = BufferedBatchIter(data)
        self.human_dataset_batch_size = self.hp.batch_size 
        self.gathered_xp_batch_size = 0

        self.evaluator = Evaluator(env)

        # store stuff used to interact with the environment here i.e. anything that 
        # would normally be a loop variable in a normal RL training script should
        # be in here.
        self.env_interaction = {
            "needs_reset": True,
            "last_state": None,
            "episode_return": 0,
        }
    
    def sample(self, strategy: callable)-> Dict[str, np.ndarray]:
        if len(self.gathered_transitions) >= strategy(self.human_dataset_batch_size, self.gathered_xp_batch_size, self.hp.sampling_step)[1]: 
            self.human_dataset_batch_size, self.gathered_xp_batch_size = \
                strategy(self.human_dataset_batch_size, self.gathered_xp_batch_size, self.hp.sampling_step)
        
        dataset_batch = self._get_dataset_batches(self.human_dataset_batch_size)[0]
        gathered_batch = self.gathered_transitions.sample(self.gathered_xp_batch_size)
        
        return ReplayBuffer.create_batch_sample(
            np.concatenate((dataset_batch['reward'], gathered_batch['reward'])),
            np.concatenate((dataset_batch['done'], gathered_batch['done'])),
            np.concatenate((dataset_batch['action'], gathered_batch['action'])),
            {key: np.concatenate(
                (dataset_batch['state'][key], gathered_batch['state'][key])
                ) for key in dataset_batch['state']},
            {key: np.concatenate(
                (dataset_batch['next_state'][key], gathered_batch['state'][key])
                ) for key in dataset_batch['next_state']}
        )


    def _get_dataset_batches(self, batch_size: int, num_batches: int = 1) -> Dict[str, np.ndarray]:
        batches = []
        for current_state, action, reward, next_state, done \
            in self.dataset_iter.buffered_batch_iter(batch_size=batch_size, num_batches=num_batches):
            batches.append(
                ReplayBuffer.create_batch_sample(
                    reward, done, action, current_state, next_state
                )
            )
        return batches


    def train(self) -> None:
        """main training function. This basic training loop should be enough for most conventional RL algorithms"""

        t = 0
        while t < self.hp.train_steps:
            log_dict = {"step": t}

            if t % self.hp.gather_every == 0:
                log_dict.update(self._gather(self.hp.gather_n))

            if t > self.hp.burn_in and t % self.hp.train_every == 0:
                log_dict.update(self._train_step(t))

            if t % self.hp.evaluate_every == 0:
                log_dict.update(self.evaluator.evaluate(
                    self.agent, self.hp.evaluate_episodes
                ))

            log_dict.update(self._housekeeping(t))

            self._log(log_dict)

    def _gather(self, steps: int) -> Dict[str, Any]:
        """gathers steps of experience from the environment

        Args:
            steps (int): how many steps of experience to gather
        """
        log_dict = {}

        start_time = time.perf_counter()

        for _ in range(steps):
            if self.env_interaction['needs_reset']:
                state = self.env.reset()
            else:
                state = self.env["last_state"]
            
            action = self.agent.act(state=state)

            next_state, reward, done, info = self.env.step(action)

            self.gathered_transitions.add(state, action, next_state, reward, done)

            self.env_interaction["episode_return"] += reward
            self.env_interaction["last_state"] = state

            if done:
                log_dict["episode_return"] = self.env_interaction["episode_return"]
                
                self.env_interaction["episode_return"] = 0
                self.env_interaction["needs_reset"] = True
                self.env["last_state"] = None
        
        end_time = time.perf_counter()

        log_dict["gather_fps"] = steps / (end_time - start_time)

        return log_dict

    @abstractmethod
    def _train_step(self, step: int) -> None:
        raise NotImplementedError()

    def _housekeeping(self, step: int) -> None:
        #TODO: implement
        return {}


    def _log(self, log_dict: dict) -> None:
        if self.use_wandb:
            wandb.log(log_dict)


# TODO: write tests
class DQNTrainer(BaseTrainer):
    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: DQNHyperparameters, use_wandb: bool = False
    ) -> None:
        super().__init__(env=env, agent=agent, hyperparameters=hyperparameters, use_wandb=use_wandb)

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=self.hp.lr)

    def _train_step(self, step: int) -> None:
        # Get a batch of experience from the gathered transitions
        batch = self.sample()

        # convert np transitions into torch tensors
        batch["state"] = torch.from_numpy(batch["state"]).to(
            self.device, dtype=torch.float32
        )
        batch["next_state"] = torch.from_numpy(batch["next_state"]).to(
            self.device, dtype=torch.float32
        )
        batch["action"] = torch.tensor(
            batch["action"], dtype=torch.long, device=self.device
        )
        batch["reward"] = torch.tensor(
            batch["reward"], dtype=torch.float32, device=self.device
        )
        batch["done"] = torch.tensor(
            batch["done"], dtype=torch.float32, device=self.device
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
        q_values = self.q1(batch["state"])
        actions = batch["action"].unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        # estimate q values for next states/actions using q2 network
        next_actions = self.q2(batch["next_state"]).argmax(dim=1).unsqueeze(1)
        next_q_values = self.q2(batch["next_state"]).gather(1, next_actions).squeeze(1)

        # calculate TD target for Bellman Equation
        td_target = self.reward_scale * batch[
            "reward"
        ] + self.gamma * next_q_values * (1 - batch["done"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target)

        return loss
