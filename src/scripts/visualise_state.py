import matplotlib.pyplot as plt
import gym
import minerl
import dataclasses

from minerl3161.utils.wrappers import minerlWrapper
from minerl3161.hyperparameters import DQNHyperparameters

def main():
    print("creating env...")
    env = gym.make('MineRLNavigateDense-v0')

    print("resetting env...")
    state = env.reset()["pov"]

    plt.figure()
    plt.imshow(state)
    plt.savefig("state.png")
    
    wrapped_env = minerlWrapper(env, **dataclasses.asdict(DQNHyperparameters()))

    state = wrapped_env.reset()['pov'][-1]

    plt.figure()
    plt.imshow(state)
    plt.savefig("wrapped_frame.png")



if __name__ == "__main__":
    main()