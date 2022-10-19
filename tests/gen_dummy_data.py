from pathlib import Path
import gym

from minerl3161.buffers import ReplayBuffer, PrioritisedReplayBuffer
from minerl3161.wrappers import cartPoleWrapper

def gen_data(n=100):
    env = cartPoleWrapper(gym.make("CartPole-v0"))
    b1 = ReplayBuffer(n, obs_space=env.observation_space)
    b2 = PrioritisedReplayBuffer(n, env.observation_space, alpha=0.2)

    d = False
    s = env.reset()

    for _ in range(n):
        a = env.action_space.sample()

        s_, r, d, _ = env.step(a)

        b1.add(s, a, s_, r, d)
        b2.add(s, a, s_, r, d)

        s = s_

        if d:
            s = env.reset()
    
    out_dir = Path(__file__).parent.joinpath("dummy_data")
    out_dir.mkdir(exist_ok=True, parents=True)

    b1.save(out_dir.joinpath("dummy_replay.pkl"))
    b2.save(out_dir.joinpath("dummy_prioritised.pkl"))


if __name__ == "__main__":
    gen_data()