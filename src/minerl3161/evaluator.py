from minerl3161.agent import BaseAgent
import gym
from minerl3161.configs import evaluator_video_path

# TODO: write tests
class Evaluator:
    def __init__(self, env) -> None:
        self.env = gym.wrappers.Monitor(env, evaluator_video_path + '/eval', force=True)
        
        self.env_interaction = {
            "needs_reset": True,
            "last_state": None,
            "episode_return": 0,
            "episode_length": 0
        }

    def evaluate(self, agent: BaseAgent, episodes: int) -> dict:
        self.env_interaction["needs_reset"] = True
        
        info = {
            'episode_return': [],
            'episode_length': []
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
                
                action, _ = agent.act(state=state, train=True, step=train_step)
                action = action.detach().cpu().numpy().item()

                next_state, reward, done, _ = self.env.step(action)

                # self.env.render()

                self.env_interaction["episode_return"] += reward
                self.env_interaction["last_state"] = next_state
                self.env_interaction["episode_length"] += 1
                

            info["episode_return"].append(self.env_interaction["episode_return"])
            info['episode_length'].append(self.env_interaction["episode_length"])
            
            self.env_interaction["episode_length"] = 0
            self.env_interaction["episode_return"] = 0
            self.env_interaction["needs_reset"] = True
            self.env_interaction["last_state"] = None

        return info

    def create_media(self, agent: BaseAgent) -> dict:
        return {}
