from pathlib import Path

import torch as th
import gym

from minerl3161.agents import BaseAgent
from minerl3161 import evaluator_video_path


class Evaluator:
    """
    Evaluator class used to integrate with the BaseTrainer class, and any of its children. Works as a callback which is
    used during training as a way to monitor the agent't performance when it chooses exploited actions only.
    """

    def __init__(self, env: gym.Env, capture_video: bool) -> None:
        """
        Initialiser for Evaluator

        Args:
            env (gym.Env): the env that the agent will be evaluated in
            capture_video (bool): specifies whether a video of the eval should be captured
        """
        out_pth = evaluator_video_path + "/eval"
        Path(out_pth).mkdir(exist_ok=True, parents=True)
        self.env = gym.wrappers.Monitor(env, out_pth, force=True, video_callable=lambda x: True) if capture_video else env
        
        self.env_interaction = {
            "needs_reset": True,
            "last_state": None,
            "eval/episode_return": 0,
            "eval/episode_length": 0
        }

    def evaluate(self, agent: BaseAgent, episodes: int) -> dict:
        """
        Handles the evaluation of the agent

        Args:
            agent (BaseAgent): the agent being evaluated
            episodes (int): the number of episodes to evaluate for
        
        Returns:
            dict: a dictionary containing info from the evaluation
        """
        self.env_interaction["needs_reset"] = True
        
        info = {
            'eval/episode_return': [],
            'eval/episode_length': []
            }
            
        for _ in range(episodes):
            done = False
            train_step = 0
            
            while not done:
                train_step += 1
                if self.env_interaction['needs_reset']:
                    state = self.env.reset()
                    self.env_interaction['needs_reset'] = False
                else:
                    state = self.env_interaction["last_state"]
                
                action, _ = agent.act(state=state, train=False)

                action = action.detach().cpu().numpy() if type(action) == th.Tensor else action

                next_state, reward, done, _ = self.env.step(action=action) 

                self.env_interaction["eval/episode_return"] += reward
                self.env_interaction["last_state"] = next_state
                self.env_interaction["eval/episode_length"] += 1
                

            info["eval/episode_return"].append(self.env_interaction["eval/episode_return"])
            info['eval/episode_length'].append(self.env_interaction["eval/episode_length"])
            
            self.env_interaction["eval/episode_length"] = 0
            self.env_interaction["eval/episode_return"] = 0
            self.env_interaction["needs_reset"] = True
            self.env_interaction["last_state"] = None
        
        # Getting average from episodes
        info["eval/episode_return"] = sum(info["eval/episode_return"])/len(info["eval/episode_return"])
        info["eval/episode_length"] = sum(info["eval/episode_length"])/len(info["eval/episode_length"])

        self.env.reset()

        return info
