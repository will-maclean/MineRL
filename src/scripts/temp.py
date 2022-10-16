import gym
from tqdm import tqdm
import minerl

from minerl3161.utils.env_reset_toggler import EnvResetToggler

# Configure environment
def _make_env():
    env = gym.make("MineRLNavigateDense-v0")
    return env
    

def main():
    episodes = 5

    env = EnvResetToggler([_make_env, _make_env])

    for t in tqdm(range(episodes)):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()

            _, _, done, _ = env.step(action)
    
    env.close()


if __name__ == "__main__":
    main()