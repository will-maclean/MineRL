from typing import Union, Dict, List, Any
from time import perf_counter
from abc import abstractmethod

import gym
import numpy as np
from tqdm import tqdm
import wandb

from minerl3161.agents import BaseAgent
from minerl3161.hyperparameters import BaseHyperparameters
from minerl3161.buffers import ReplayBuffer
from minerl3161.utils.termination import TerminationCondition
from minerl3161.utils.checkpointer import Checkpointer
from minerl3161.utils.evaluator import Evaluator


class BaseTrainer:
    """
    Abstract class for Trainers. At the least, all implementations must have _train_step().
    """

    def __init__(
        self, 
        env: gym.Env, 
        agent: BaseAgent, 
        hyperparameters: BaseHyperparameters, 
        human_dataset: Union[ReplayBuffer, None] = None, 
        use_wandb: bool = False,
        device: str = "cpu", 
        render: bool = False, 
        replay_buffer_class: ReplayBuffer = ReplayBuffer, 
        replay_buffer_kwargs: Dict = {}, 
        termination_conditions: Union[List[TerminationCondition], None] = None,
        capture_eval_video: bool = True
    ) -> None:
        """
        Initialiser for BaseTrainer

        Args:
            env (gym.Env): environmnet to train in
            agent (BaseAgent): agent to train
            hyperparameters (BaseHyperparameters): hyperparameters to train with
            human_dataset (Union[ReplayBuffer, None]): a ReplayBuffer containing expert human transitions
            use_wandb (bool): dictates whether wandb should be used or not
            device (str): dictates what device the tensors should be loaded onto for training the model 
            render (bool): dictates whether the envrionment should be rendered during training or not
            replay_buffer_class (ReplayBuffer): the ReplayBuffer being used to store the transitions during training
            replay_buffer_kwargs (Dict): the kwargs that will be passed into the ReplayBuffer initialiser
            termination_conditions (Union[List[TerminationCondition], None]): the conditions that dictate when training should conclude
            capture_eval_video (bool): dictates whether a video should be captured when performing the eval callback
        """
        self.env: gym.Env = env
        self.agent: BaseAgent = agent
        self.hp: BaseHyperparameters = hyperparameters
        self.use_wandb = use_wandb
        self.device = device
        self.render = render

        if termination_conditions is not None:
            if type(termination_conditions) != list:
                termination_conditions = [termination_conditions]
            
            self.termination_conditions = termination_conditions
        else:
            self.termination_conditions = None

        self.checkpointer = Checkpointer(agent, checkpoint_every=self.hp.checkpoint_every, use_wandb=use_wandb)

        self.gathered_transitions = replay_buffer_class(
            self.hp.buffer_size_gathered, self.env.observation_space, **replay_buffer_kwargs
        )
        self.human_transitions = human_dataset

        self.human_dataset_batch_size = self.hp.batch_size 
        self.gathered_xp_batch_size = 0

        self.evaluator = Evaluator(env, capture_video=capture_eval_video)

        # store stuff used to interact with the environment here i.e. anything that 
        # would normally be a loop variable in a normal RL training script should
        # be in here.
        self.env_interaction = {
            "needs_reset": True,
            "last_state": None,
            "episode_return": 0,
            "episode_length": 0,
        }

        self.training = False
    
    def sample(self, strategy: callable) -> Dict[str, np.ndarray]:
        """
        Used to retrieve a batch of samples from the ReplayBuffer for training the model weights

        Args:
            strategy (callable): a function which is used to determine the gathered/human data split in the batch
        
        Returns:
            Dict[str, np.ndarray]: a dictionary containing the sample data
        """
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
        """
        Main training function. This basic training loop should be enough for most conventional RL algorithms.
        """
        self.training = True  # flag that lets termination conditions stop training

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
                self.env_interaction['needs_reset'] = True

            log_dict.update(self._housekeeping(t))

            self._log(log_dict)

            if not self.training:
                break
        
        self.close()

    @abstractmethod
    def _train_step(self, step: int) -> Dict[str, np.ndarray]:
        """
        Abstract method which all subclasses MUST implement. Determines how the model is trained.

        Args:
            step (int): the current time step in the training
        """
        raise NotImplementedError()

    def _gather(self, steps: int) -> Dict[str, Any]:
        """
        Gathers steps of experience from the environment

        Args:
            steps (int): how many steps of experience to gather
        
        Returns:
            Dict[str, Any]: a dictionary containing data for logging
        """
        log_dict = {}

        start_time = perf_counter()

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

            if self.render:
                self.env.render()

            self.add_transition(state, action, next_state, reward, done)
            self.env_interaction["episode_length"] += 1
            self.env_interaction["episode_return"] += reward
            self.env_interaction["last_state"] = next_state

            if done:
                log_dict["episode_return"] = self.env_interaction["episode_return"]
                log_dict["episode_length"] = self.env_interaction["episode_length"]

                if self.termination_conditions is not None:
                    self._process_termination_conditions(self.env_interaction)

                self.env_interaction["episode_return"] = 0
                self.env_interaction["needs_reset"] = True
                self.env_interaction["last_state"] = None
                self.env_interaction["episode_length"] = 0
        
        end_time = perf_counter()

        log_dict["gather_fps"] = steps / (end_time - start_time)

        return log_dict
    
    def add_transition(
        self, 
        state: Dict[str, np.ndarray],
        action: Union[np.ndarray, float],
        next_state: Dict[str, np.ndarray],
        reward: Union[np.ndarray, float],
        done: Union[np.ndarray, bool],
    ) -> None:
        """
        Used to add a transition to the ReplayBuffer

        Args:
            state (Dict[str, np.ndarray]): the environment state at the given time step
            action (Union[np.ndarray, float]): the action taken in the envrionment at the given time step
            next_state (Dict[str, np.ndarray]): the environment state the agent ends up in after taking the action
            reward (Union[np.ndarray, float]): the reward obtained from performing the action
            done (Union[np.ndarray, bool]): a flag that represents whether or not the taken action ended the current episode
        """
        self.gathered_transitions.add(state, action, next_state, reward, done)

    def _housekeeping(self, step: int) -> Dict[str, Any]:
        """
        Used to update the log dictionary

        Args:
            step (int): the current time step in training

        Returns:
            Dict[str, Any]: the dictionary used for logging
        """
        log_dict = {}

        # start with checkpointing
        log_dict.update(
            self.checkpointer.step(step)
        )
        
        return log_dict
    
    def _process_termination_conditions(self, env_interation: Dict[str, Any]) -> None:
        """
        Goes through all of the termination conditions with the env_interaction data, to see if training should
        be terminated
        
        Args:
            env_interation (Dict[str, Any]): the dictionary of episode interaction data
        """
        terminate_training = False
        for t in self.termination_conditions:
            terminate_training = terminate_training or t(**env_interation)
        
        if terminate_training:
            self.training = False

    def _log(self, log_dict: dict) -> None:
        """
        Used to log the contents of the log dictionary to wandb

        Args:
            log_dict (dict): the dictionary that contains the logged data
        """
        if self.use_wandb:
            wandb.log(log_dict)
    
    def close(self) -> None:
        """
        Gets called at the end of training
        """
        pass
