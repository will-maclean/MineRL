import os
from pathlib import Path

import wandb
from minerl3161.agent import BaseAgent
import gym
from minerl3161.configs import evaluator_video_path

# TODO: write tests
class Evaluator:
    def __init__(self, env, use_wandb=True) -> None:
        out_pth = evaluator_video_path + "eval"
        Path(out_pth).mkdir(exist_ok=True, parents=True)
        self.env = gym.wrappers.Monitor(env, out_pth, force=True)
        
        self.env_interaction = {
            "needs_reset": True,
            "last_state": None,
            "eval/episode_return": 0,
            "eval/episode_length": 0
        }

    def evaluate(self, agent: BaseAgent, episodes: int) -> dict:
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

                action = action.detach().cpu().numpy()

                next_state, reward, done, _ = self.env.step(action=action) 

                # self.env.render()

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

        return info

    def create_media(self, agent: BaseAgent) -> dict:
        return {}
