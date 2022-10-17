from typing import Tuple, Union, List, Dict

import gym
import torch as th
import numpy as np
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_

from minerl3161.agents import BaseAgent
from minerl3161.hyperparameters import RainbowDQNHyperparameters
from minerl3161.buffers import ReplayBuffer, PrioritisedReplayBuffer, NStepReplayBuffer
from minerl3161.utils.termination import TerminationCondition
from minerl3161.utils import np_batch_to_tensor_batch, copy_weights, linear_decay

from .base_trainer import BaseTrainer


class RainbowDQNTrainer(BaseTrainer):
    """
    The Trainer for the RainbowDQNAgent. Inherits from the BaseTrainer class, implementing the _train_step() method.
    """

    def __init__(
        self, 
        env: gym.Env, 
        agent: BaseAgent, 
        hyperparameters: RainbowDQNHyperparameters, 
        human_dataset: Union[PrioritisedReplayBuffer, None] = None, 
        use_wandb: bool = False, 
        device: str = "cpu", 
        render: bool = False,
        termination_conditions: Union[List[TerminationCondition], None] = None,
        capture_eval_video: bool = True
    ) -> None:
        """
        Initialiser for RainbowDQNTrainer

        Args:
            env (gym.Env): environmnet to train in
            agent (BaseAgent): agent to train
            hyperparameters (RainbowDQNHyperparameters): hyperparameters to train with
            human_dataset (Union[PrioritisedReplayBuffer, None]): a PrioritisedReplayBuffer containing expert human transitions
            use_wandb (bool): dictates whether wandb should be used or not
            device (str): dictates what device the tensors should be loaded onto for training the model 
            render (bool): dictates whether the envrionment should be rendered during training or not
            termination_conditions (Union[List[TerminationCondition], None]): the conditions that dictate when training should conclude
            capture_eval_video (bool): dictates whether a video should be captured when performing the eval callback
        """
        super().__init__(env=env, 
            agent=agent,
            human_dataset=human_dataset, 
            hyperparameters=hyperparameters, 
            use_wandb=use_wandb, 
            device=device, 
            replay_buffer_class=PrioritisedReplayBuffer, 
            replay_buffer_kwargs={"alpha": hyperparameters.alpha}, 
            render=render,
            termination_conditions=termination_conditions,
            capture_eval_video=capture_eval_video
        )

        # The optimiser keeps track of the model weights that we want to train
        self.optim = Adam(self.agent.q1.parameters(), lr=hyperparameters.lr)

        # memory for N-step Learning
        self.use_n_step = True if hyperparameters.n_step > 1 else False
        if self.use_n_step:
            self.n_step = hyperparameters.n_step
            self.memory_n = NStepReplayBuffer(
                n=hyperparameters.buffer_size_gathered, 
                hyperparameters=hyperparameters 
            )
        
        self.beta = hyperparameters.beta_min
        self.prior_eps = hyperparameters.prior_eps

        # Categorical DQN parameters
        self.v_min = hyperparameters.v_min
        self.v_max = hyperparameters.v_max
        self.atom_size = hyperparameters.atom_size
    
    def add_transition(
        self, 
        state, 
        action, 
        next_state, 
        reward, 
        done
    ) -> None:
        """
        Used to add a transition to the PrioritisedReplayBuffer

        # TODO: licence

        Args:
            TODO
        """
        transition = (state, action, next_state, reward, done)
        
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.add(*transition)
        # 1-step transition
        else:
            one_step_transition = transition

        # add a single step transition
        if one_step_transition:
            self.gathered_transitions.add(*one_step_transition)

    def _train_step(self, step: int) -> None:
        """
        Implements the network training that is to be completed at each train step

        TODO: licence

        Args:
            step (int): the current time step in the training
        """
        log_dict = {}
        # Get a batch of experience from the gathered transitions
        batch, sample_log_dict = self.sample(step)

        log_dict.update(sample_log_dict)

        # convert np transitions into torch tensors
        batch = np_batch_to_tensor_batch(batch, self.device)
        batch["weights"] = th.tensor(
            batch["weights"].reshape(-1, 1), dtype=th.float32, device=self.device
        )

        # 1-step Learning loss
        elementwise_loss = self._calc_loss(batch)

        # PER: importance sampling before average
        loss = th.mean(elementwise_loss * batch["weights"])

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.hp.gamma ** self.n_step
            sample = self.memory_n.sample_batch_from_idxs(self.gathered_transitions, batch["indices"])
            sample = np_batch_to_tensor_batch(sample, self.device)
            elementwise_loss_n_loss = self._calc_loss(sample, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = th.mean(elementwise_loss * batch["weights"])

        # update model parameters
        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.agent.q1.parameters(), 10.0)
        self.optim.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.gathered_transitions.update_priorities(batch["indices"], new_priorities)
        
        # NoisyNet: reset noise
        self.agent.q1.reset_noise()
        self.agent.q2.reset_noise()

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

    def _calc_loss(self, batch: dict, gamma: float = None) -> th.Tensor:
        """
        Used to calculate the loss of the supplied batch (the margin of error between the q-values that model currently predicts, and
        the newly calculated ones)

        Args:
            batch (dict): the batch of data extracted from the ReplayBuffer, being used to train the data
            gamma (float): hyperparameter used to ensure the infinite Bellman Equation converges onto a number (known as the discount factor)
        
        Returns:
            th.Tensor: the average loss of each of the samples in the batch
        """
        gamma = self.hp.gamma if gamma is None else gamma

        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with th.no_grad():
            # Double DQN
            next_action = self.agent.q1(next_state).argmax(1)
            next_dist = self.agent.q2.dist(next_state)
            next_dist = next_dist[range(self.hp.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.agent.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                th.linspace(
                    0, (self.hp.batch_size - 1) * self.atom_size, self.hp.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.hp.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = th.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.agent.q1.dist(state)
        log_p = th.log(dist[range(self.hp.batch_size), action.squeeze(1)])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    
    def sample(self, step: int) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Used to retrieve a batch of samples from the ReplayBuffer for training the model weights

        Args:
            step (int): the current time step in training
        
        Returns:
            Tuple[Dict[str, np.ndarray], dict]: a dictionary containing the sample data, along with the logging dictionary
        """
        beta = linear_decay(step=step, start_val=self.hp.beta_min, final_val=self.hp.beta_max, final_steps=self.hp.beta_final_step)
        return_batch, gathered_weights, gathered_indices =  self.gathered_transitions.sample(self.hp.batch_size, beta=beta)
        return_batch["weights"] = gathered_weights
        return_batch["indices"] = gathered_indices

        return return_batch, {"beta": beta}
