""" BaseTrainer and implementations stored here
"""
from abc import abstractmethod
from typing import Any, Dict, Union
import time

import gym
import numpy as np
import wandb
from tqdm import tqdm
import torch
from minerl3161.checkpointer import Checkpointer
from minerl3161.utils import copy_weights
from torch.optim import Adam

from minerl3161.agent import BaseAgent
from minerl3161.buffer import ReplayBuffer, PrioritisedReplayBuffer
from minerl3161.evaluator import Evaluator
from minerl3161.hyperparameters import BaseHyperparameters, DQNHyperparameters
from minerl3161.utils import np_dict_to_pt, linear_decay


# TODO: write tests
class BaseTrainer:
    """Abstract class for Trainers. At the least, all implementations must have _train_step()."""

    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: BaseHyperparameters, human_dataset: Union[ReplayBuffer, None] = None, use_wandb: bool =False,
        device="cpu", replay_buffer_class=ReplayBuffer, replay_buffer_kwargs={},
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
        self.device = device

        self.checkpointer = Checkpointer(agent, checkpoint_every=self.hp.checkpoint_every, use_wandb=use_wandb)

        self.gathered_transitions = replay_buffer_class(
            self.hp.buffer_size_gathered, self.env.observation_space, **replay_buffer_kwargs
        )
        self.human_transitions = human_dataset

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
            "episode_length": 0,
        }
    
    def sample(self, strategy: callable)-> Dict[str, np.ndarray]:
        if self.human_transitions is None:
            return self.gathered_transitions.sample(self.hp.batch_size)
        
        human_dataset_batch_size, gathered_xp_batch_size \
            = strategy(self.hp.batch_size, 
                        step=self.t, 
                        start_val=self.hp.sample_max, 
                        final_val=self.hp.sample_min, 
                        final_steps=self.hp.sample_final_step)
        
        if len(self.gathered_transitions) < gathered_xp_batch_size:
            gathered_xp_batch_size = len(self.gathered_transitions)
            human_dataset_batch_size = self.hp.batch_size - self.gathered_xp_batch_size
        
        self.human_dataset_batch_size = human_dataset_batch_size
        self.gathered_xp_batch_size = gathered_xp_batch_size
        
        human_batch = self.human_transitions.sample(self.human_dataset_batch_size)
        gathered_batch = self.gathered_transitions.sample(self.gathered_xp_batch_size)

        if gathered_batch['reward'].size == 0:
            return ReplayBuffer.create_batch_sample(
                human_batch['reward'],
                human_batch['done'],
                human_batch['action'],
                human_batch['state'],
                human_batch['next_state']
            )
        
        return ReplayBuffer.create_batch_sample(
            np.concatenate((human_batch['reward'], gathered_batch['reward'])),
            np.concatenate((human_batch['done'], gathered_batch['done'])),
            np.concatenate((human_batch['action'], gathered_batch['action'])),
            {key: np.concatenate(
                (human_batch['state'][key], gathered_batch['state'][key])
                ) for key in human_batch['state']},
            {key: np.concatenate(
                (human_batch['next_state'][key], gathered_batch['state'][key])
                ) for key in human_batch['next_state']}
        )


    def train(self) -> None:
        """main training function. This basic training loop should be enough for most conventional RL algorithms"""

        for t in tqdm(range(self.hp.train_steps)):
            self.t = t

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
        
        self.close()

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
                self.env_interaction['needs_reset'] = False
            else:
                state = self.env_interaction["last_state"]
            
            action, act_log_dict = self.agent.act(state=state, train=True, step=self.t)

            action = action.detach().cpu().numpy().item()

            log_dict.update(act_log_dict)

            next_state, reward, done, info = self.env.step(action)

            # self.env.render()

            self.gathered_transitions.add(state, action, next_state, reward, done)
            self.env_interaction["episode_length"] += 1
            self.env_interaction["episode_return"] += reward
            self.env_interaction["last_state"] = next_state

            if done:
                log_dict["episode_return"] = self.env_interaction["episode_return"]
                
                self.env_interaction["episode_return"] = 0
                self.env_interaction["needs_reset"] = True
                self.env_interaction["last_state"] = None
                self.env_interaction["episode_length"] = 0
        
        end_time = time.perf_counter()

        log_dict["gather_fps"] = 1 / (end_time - start_time)

        return log_dict

    @abstractmethod
    def _train_step(self, step: int) -> None:
        raise NotImplementedError()

    def _housekeeping(self, step: int) -> None:
        log_dict = {}

        # start with checkpointing
        log_dict.update(
            self.checkpointer.step(step)
        )
        
        return log_dict


    def _log(self, log_dict: dict) -> None:
        if self.use_wandb:
            wandb.log(log_dict)
    
    def close(self):
        pass

# TODO: write tests
class DQNTrainer(BaseTrainer):
    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: DQNHyperparameters, human_dataset: ReplayBuffer, use_wandb: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__(env=env, agent=agent, human_dataset=human_dataset, hyperparameters=hyperparameters, use_wandb=use_wandb, device=device)

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=self.hp.lr)

    def _train_step(self, step: int) -> None:
        log_dict = {}
        start_time = time.perf_counter()

        # Get a batch of experience from the gathered transitions
        batch = self.sample(self.hp.sampling_strategy)

        # convert np transitions into torch tensors
        batch["state"] = np_dict_to_pt(batch["state"], device=self.device)

        batch["next_state"] = np_dict_to_pt(batch["next_state"], device=self.device)

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
            self.hp.hard_update_freq != 0
            and step % self.hp.hard_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=None)
        if (
            self.hp.soft_update_freq != 0
            and step % self.hp.soft_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=self.hp.polyak_tau)
        
        end_time = time.perf_counter()
        log_dict["train_fps"] = steps / (end_time - start_time)

        return {"loss": loss.detach().cpu().item()}

    def _calc_loss(self, batch: dict) -> torch.Tensor:
        # estimate q values for current states/actions using q1 network
        q_values = self.agent.q1(batch["state"])
        q_values = q_values.gather(1, batch["action"])

        # estimate q values for next states/actions using q2 network

        next_q_values = self.agent.q2(batch["next_state"])
        next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_q_values = next_q_values.gather(1, next_actions)

        # calculate TD target for Bellman Equation
        td_target = self.hp.reward_scale * torch.sign(batch[
            "reward"
        ]) + self.hp.gamma * next_q_values * (1 - batch["done"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target)

        return loss
    

class RainbowDQNTrainer(BaseTrainer):
    def __init__(
        self, env: gym.Env, agent: BaseAgent, hyperparameters: DQNHyperparameters, human_dataset: PrioritisedReplayBuffer, use_wandb: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__(env=env, agent=agent, human_dataset=human_dataset, hyperparameters=hyperparameters, use_wandb=use_wandb, device=device, replay_buffer_class=PrioritisedReplayBuffer, replay_buffer_kwargs={"alpha": hyperparameters.alpha})

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=self.hp.lr)

    def _train_step(self, step: int) -> None:
        log_dict = {}
        # Get a batch of experience from the gathered transitions
        batch, sample_log_dict = self.sample(self.hp.sampling_strategy, step)

        log_dict.update(sample_log_dict)

        # convert np transitions into torch tensors
        batch["state"] = np_dict_to_pt(batch["state"], device=self.device)

        batch["next_state"] = np_dict_to_pt(batch["next_state"], device=self.device)

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
            self.hp.hard_update_freq != 0
            and step % self.hp.hard_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=None)
        if (
            self.hp.soft_update_freq != 0
            and step % self.hp.soft_update_freq == 0
        ):
            copy_weights(copy_from=self.agent.q1, copy_to=self.agent.q2, polyak=self.hp.polyak_tau)

        log_dict["loss"] = loss.detach().cpu().item()

        return log_dict

    def _calc_loss(self, batch: dict) -> torch.Tensor:
        gathered_batch_size = batch["gathered_batch_size"]
        human_batch_size = batch["human_batch_size"]

        human_weights = torch.FloatTensor(
            batch["human_weights"].reshape(-1, 1)
        ).to(self.device)
        gathered_weights = torch.FloatTensor(
            batch["gathered_weights"].reshape(-1, 1)
        ).to(self.device)
        
        human_indices = batch["human_indices"]
        gathered_indices = batch["gathered_indices"]
        
        # estimate q values for current states/actions using q1 network
        q_values = self.agent.q1(batch["state"])
        q_values = q_values.gather(1, batch["action"])

        # estimate q values for next states/actions using q2 network

        next_q_values = self.agent.q2(batch["next_state"])
        next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_q_values = next_q_values.gather(1, next_actions)

        # calculate TD target for Bellman Equation
        td_target = self.hp.reward_scale * torch.sign(batch[
            "reward"
        ]) + self.hp.gamma * next_q_values * (1 - batch["done"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = torch.nn.functional.smooth_l1_loss(q_values, td_target, reduction="none")

        # update the weights
        loss_for_prior = loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hp.prior_eps

        # need to split the new priorities
        n_human = human_weights.shape[0]
        new_priorities_human = new_priorities[:n_human]  # human experience must be first in the batch for this to work!!
        new_priorities_gathered = new_priorities[n_human:]

        if human_batch_size > 0 :
            self.human_transitions.update_priorities(human_indices, new_priorities_human)
        
        if gathered_batch_size > 0 :
            self.gathered_transitions.update_priorities(gathered_indices, new_priorities_gathered)
        
        if human_batch_size == 0:
            weights = gathered_weights
        elif gathered_batch_size == 0:
            weights = human_weights
        else:
            weights = torch.concat([human_weights, gathered_weights], dim=0)

        loss = loss * weights

        return loss.mean()
    
    def sample(self, strategy: callable, step)-> Dict[str, np.ndarray]:
        human_dataset_batch_size, gathered_xp_batch_size \
            = strategy(batch_size=self.hp.batch_size, 
                        step=step, 
                        start_val=self.hp.sample_max, 
                        final_val=self.hp.sample_min, 
                        final_steps=self.hp.sample_final_step)
        
        if len(self.gathered_transitions) < gathered_xp_batch_size:
            gathered_xp_batch_size = len(self.gathered_transitions)
            human_dataset_batch_size = self.hp.batch_size - self.gathered_xp_batch_size
        
        self.human_dataset_batch_size = human_dataset_batch_size
        self.gathered_xp_batch_size = gathered_xp_batch_size
        
        beta = linear_decay(step=step, start_val=self.hp.beta_max, final_val=self.hp.beta_min, final_steps=self.hp.sample_final_step)

        gathered_weights = np.empty((1,))
        gathered_indices = np.empty((1,))
        human_weights = np.empty((1,))
        human_indices = np.empty((1,))

        if self.human_dataset_batch_size == 0:
            return_batch, gathered_weights, gathered_indices = self.gathered_transitions.sample(self.gathered_xp_batch_size, beta)
        
        elif self.gathered_xp_batch_size == 0:
            return_batch, human_weights, human_indices = self.human_transitions.sample(self.human_dataset_batch_size, beta)
        
        else:
            human_batch, human_weights, human_indices = self.human_transitions.sample(self.human_dataset_batch_size, beta)
            gathered_batch, gathered_weights, gathered_indices = self.gathered_transitions.sample(self.gathered_xp_batch_size, beta)

            return_batch =  ReplayBuffer.create_batch_sample(
                np.concatenate((human_batch['reward'], gathered_batch['reward'])),
                np.concatenate((human_batch['done'], gathered_batch['done'])),
                np.concatenate((human_batch['action'], gathered_batch['action'])),
                {key: np.concatenate(
                    (human_batch['state'][key], gathered_batch['state'][key])
                    ) for key in human_batch['state']},
                {key: np.concatenate(
                    (human_batch['next_state'][key], gathered_batch['state'][key])
                    ) for key in human_batch['next_state']}
            )
        
        return_batch["gathered_weights"] = gathered_weights
        return_batch["gathered_indices"] = gathered_indices
        return_batch["human_weights"] = human_weights
        return_batch["human_indices"] = human_indices
        return_batch["gathered_batch_size"] = self.gathered_xp_batch_size
        return_batch["human_batch_size"] = self.human_dataset_batch_size

        return return_batch, {"beta": beta, "human_dataset_batch_size": self.human_dataset_batch_size, "gathered_xp_batch_size": self.gathered_xp_batch_size}
