import gym
import wandb

from agent import DQNAgent

env = gym.make("CartPole-v0")

wandb.init(
    project="CartPoleRainbowTEST", 
    entity="minerl3161",
)

agent = DQNAgent(
    env=env, 
    memory_size=50000, 
    batch_size=16, 
    target_update=5000,
    n_step=1
)

agent.train(
    50000, 
    plot_title="Test Run",
    stopper=None
)