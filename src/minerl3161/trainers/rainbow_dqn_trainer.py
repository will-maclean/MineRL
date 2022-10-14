from typing import Union, List, Dict

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
                size=hyperparameters.buffer_size_gathered, 
                batch_size=hyperparameters.batch_size, 
                gamma=hyperparameters.gamma,
                n_step=hyperparameters.n_step, 
            )
        
        self.beta = hyperparameters.beta_min
        self.prior_eps = hyperparameters.prior_eps

        # Categorical DQN parameters
        self.v_min = hyperparameters.v_min
        self.v_max = hyperparameters.v_max
        self.atom_size = hyperparameters.atom_size
    
    def add_transition(self, state, action, next_state, reward, done):
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
        log_dict = {}
        # Get a batch of experience from the gathered transitions
        batch, sample_log_dict = self.sample(self.hp.sampling_strategy, step)

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
        gamma = self.hp.gamma if gamma is None else gamma
        
        if self.human_transitions is None:
            return self._calc_loss_gathered_only(batch, gamma)
        else:
            return self._calc_loss_combined(batch)

    def _calc_loss_gathered_only(self, batch, gamma):
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

    def _calc_loss_combined(self, batch):
        gathered_batch_size = batch["gathered_batch_size"]
        human_batch_size = batch["human_batch_size"]

        human_weights = th.FloatTensor(
            batch["human_weights"].reshape(-1, 1)
        ).to(self.device)
        gathered_weights = th.FloatTensor(
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
        td_target = self.hp.reward_scale * th.sign(batch[
            "reward"
        ]) + self.hp.gamma * next_q_values * (1 - batch["done"])

        # Calculate loss for Bellman Equation
        # Note that we use smooth_l1_loss instead of MSE as it is more stable for larger loss signals. RL problems
        # typically suffer from high variance, so anything we can do to introduce more stability is usually a
        # good thing.
        loss = th.nn.functional.smooth_l1_loss(q_values, td_target, reduction="none")

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
            weights = th.concat([human_weights, gathered_weights], dim=0)

        loss = loss * weights

        return loss.mean()
    
    def sample(self, strategy: callable, step)-> Dict[str, np.ndarray]:
        
        # if we have not human transitions, then yeah obviously we can simplify this provess a bit
        if self.human_transitions is None:
            beta = linear_decay(step=step, start_val=self.hp.beta_min, final_val=self.hp.beta_max, final_steps=self.hp.beta_final_step)
            return_batch, gathered_weights, gathered_indices =  self.gathered_transitions.sample(self.hp.batch_size, beta=beta)
            return_batch["weights"] = gathered_weights
            return_batch["indices"] = gathered_indices

            return return_batch, {"beta": beta}
        
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
        
        beta = linear_decay(step=step, start_val=self.hp.beta_min, final_val=self.hp.beta_max, final_steps=self.hp.beta_final_step)

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