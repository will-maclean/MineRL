from typing import Dict, List, Tuple

import gym
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
#from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import wandb
import h5py

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from network import Network
from utils import Stopper
from dataloader import Dataloader


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        lr: float = 2e-4,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        noisy_init: float = 0.1,
        folder_name: str = "env",
        plot_title: str = "plot",
        # Massive
        massive: bool = False,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.burn_in = 500
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading onto {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

        # Massive
        self.massive = massive
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support, noisy_init
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support, noisy_init
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        self.folder_name = folder_name
        self.plot_title = plot_title

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray, episode_reward: float, episode_time_steps: int) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # next_state, reward, done, _ = self.env.step(action, episode_reward, episode_time_steps)
        next_state, reward, done, _ = self.env.step(action) # changed to work for mountaincar

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_frames: int, stopper: Stopper, plot_title: str, reward_func: object = None, weights_filename: str = "model"):
        """
        num_frames: how many time steps the agent will train
        plotting_interval: the data should be plotted every plotting_interval time steps
        reward_limit: the maximum reward the agent can get, in the time step when this reward is met or exceeded, the episode is ended
        recent_n_running_avg: represents the number of time steps contrubting to the running average (i.e., n last time steps)
        """
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        # state = self.env.reset() # Changed to work for mountaincar
        update_cnt = 0
        losses = []
        scores = []
        t_steps = []
        score = 0
        episode = 0
        time_steps = 0
        episode_time_steps = 0

        start = time.time()

        for frame_idx in tqdm(range(1, num_frames + 1)):
            episode_time_steps += 1
            action = self.select_action(state)
            next_state, reward, done = self.step(action, score, episode_time_steps)

            # REWARD FUNCTION VARIATION
            if reward_func is not None:
                reward = reward_func(reward, episode_time_steps)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state = self.env.reset()
                # state = self.env.reset() # changed to work for mountaincar
                scores.append(score)
                # stopper.update(score)

                if len(losses) > 0:
                    # self._plot(frame_idx, scores, losses, "test")
                    wandb.log({
                        f'BC Benchmark - Training Graphs/{plot_title}/Score': score,
                        f'BC Benchmark - Training Graphs/{plot_title}/Loss': losses[-1],
                        f'BC Benchmark - Training Graphs/{plot_title}/Episode': episode
                    })
                
                time_steps += episode_time_steps
                t_steps.append(episode_time_steps)
                episode_time_steps = 0

                episode += 1
                score = 0
                

            # if training is ready
            if self.memory.tree_ptr >= self.burn_in:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # self._soft_target_update(1e-5)

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
            
            # if stopper.check_cond():
            #     torch.save(self.dqn.state_dict(), f"models/{self.folder_name}/{weights_filename}.pt")
            #     print(f"Training finished after stop condition was consistenly met {stopper.size} times.")
            #     break
                        
              
        print(f"Training time: {str(datetime.timedelta(seconds=time.time() - start)).split('.')[0]}")
        print(f"Time steps: {time_steps}")

        print(f"Saving model weights to {weights_filename}.pt")
        torch.save(self.dqn.state_dict(), f"models/{self.folder_name}/{weights_filename}.pt")
        self.env.close()
    
    def evaluate(self, time_steps: int, plot_title: str, reward_goal: int = 500, episode_count: int = None, reward_func: object = None, weights_filename: str = "model", save_as: str = None, save_data: bool = True):
        print(f"Evaluating {weights_filename} for {time_steps} time steps...")
        
        self.dqn.load_state_dict(torch.load(f"models/{self.folder_name}/{weights_filename}.pt",map_location=self.device))
        # Load weights back to our model and set it to evaluation mode
        self.dqn.eval()

        t_steps = 0
        game = 0
        scores = []
        if save_data:
            pos_x_eps = [None] * 200 # Each index contains an entire episodes x coords in order
            pos_y_eps = [None] * 200 # Each index contains an entire episodes y coords in order

            seed = np.ones(200, dtype='int64')
            for i in range(len(seed)):
                seed[i] = i

        while t_steps <= time_steps:
            # Rinse and repeat the usual stuff.
            if save_data:
                state = self.env.reset(seed[game])
                # state = self.env.reset() # Changed to work for mountaincar
            else:
                state = self.env.reset(None)
                # state = self.env.reset() # Changed to work for mountaincar
            score = 0
            done = 0
            t = 0

            if save_data:
                pos_x = []  
                pos_y = []

                pos_x.append(state[0]) # state[0] contains the x position of the lander
                pos_y.append(state[1]) # state[1] contains the y position of the lander

            states, acts, rewards, next_states, dones = [], [], [], [], []

            while not done:
                t += 1

                if not self.massive:
                    self.env.render()

                # Evaluate the action from the model
                action = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action, score, t)
                # next_state, reward, done, _ = self.env.step(action, t) # changed to work for mountaincar

                if save_data:
                    pos_x.append(next_state[0])
                    pos_y.append(next_state[1])

                # REWARD FUNCTION VARIATION
                if reward_func is not None:
                    reward = reward_func(reward, t)

                states.append(state)
                acts.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                score += reward
            
            scores.append(score)

            wandb.log({
                f'BC Benchmark - Evaluation Graphs/{plot_title}/Score': score, 
                f'BC Benchmark - Evaluation Graphs/{plot_title}/Episode': game
              })
            
            output_string = f"{game=}, {score=}, {t_steps=}"

            if score >= reward_goal:
                # If episode score was above solved amount, add time steps to memory buffer
                for i in range(len(states)):
                    _ = self.memory.store(states[i], acts[i], rewards[i], next_states[i], dones[i])
                t_steps += t   
            else:
                output_string += f" - NOT SAVING: less than goal of: {reward_goal}"
            
            if save_data:
                pos_x_eps[game] = pos_x
                pos_y_eps[game] = pos_y

            game += 1
            print(output_string)     

            if episode_count is not None and game == episode_count:
                break

            if game >= 100:
                break
        if save_data:
            print(f"Saving {t_steps} time steps...")
            print(f"Saving evaluation data to {save_as if save_as is not None else weights_filename}.h5")
            Dataloader.save_to_h5(self.memory, file_name=save_as if save_as is not None else weights_filename, folder_name=self.folder_name)

            file_name = save_as if save_as is not None else weights_filename
            with open(f'data/{self.folder_name}/{file_name}_xy.npy', 'wb') as f:
                np.save(f, np.array(pos_x_eps, dtype="object"))
                np.save(f, np.array(pos_y_eps, dtype="object"))

        self.env.close()

        scores = np.array(scores)
        mean_scores = np.mean(scores)
        success_fail_rate = len(scores[scores > 200]) / len(scores)
        std_scores = np.std(scores[scores > 200])
        return mean_scores, success_fail_rate, std_scores

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def _soft_target_update(self,
        tau: float
        ):
        for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
        plot_filename: str,
        training: bool = True
    ):
        """Plot the training progresses."""
        #clear_output(True)

        if training:
            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.title(self.plot_title)
            plt.plot(scores)
            plt.subplot(122)
            plt.title('loss')
            plt.plot(losses)
            plt.savefig(f"figures/{self.folder_name}/training/{plot_filename}.png")
            plt.savefig(f"figures/{self.folder_name}/eval/{plot_filename}.png")
            
        else:
            plt.figure(figsize=(10, 5))
            plt.title(self.plot_title)
            plt.plot(scores)
            plt.plot(losses)
            plt.savefig(f"figures/{self.folder_name}/eval/{plot_filename}.png")

        plt.close()
        
