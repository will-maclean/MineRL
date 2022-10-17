import argparse
import dataclasses
from json import load

import gym
import minerl
import matplotlib.pyplot as plt

from minerl3161.agents import DQNAgent
from minerl3161.utils.wrappers import minerlWrapper
from minerl3161.utils.utils import np_dict_to_pt


def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", type=str, default="ckpt_1200000.zip")

    return parser.parse_args()


def main():
    args = opt()

    agent = DQNAgent.load(args.load_path)

    env = gym.make("MineRLNavigateDense-v0")
    env = minerlWrapper(env, **dataclasses.asdict(agent.hp))


    s = env.reset()
    s_, r, d, _ = env.step(env.action_space.sample())
        
    
    pixels = env.render(mode="rgb_array")

    plt.figure()
    plt.imshow(pixels)
    plt.savefig("pixels.png")

    s_pt = np_dict_to_pt(s, unsqueeze=True, device="cuda:0")

    q = agent.q1(s_pt).squeeze(0).detach().cpu().numpy()

    plt.figure()
    plt.plot(q)
    plt.savefig("q_vals.png")

    print("compass before" + str(s["compass"] * 180))

    s, r, d, i = env.step(q.argmax())

    print("compass after" + str(s["compass"] * 180))
    print(f"reward: {r}")

    


if __name__ == "__main__":
    main()
