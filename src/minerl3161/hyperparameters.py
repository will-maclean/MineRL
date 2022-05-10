from dataclasses import dataclass

@dataclass
class BaseHyperparameters:
    time_steps: int = 100000                # number of environment interaction steps
    batch_size: int = 32                    # batch size for training
    buffer_size_gathered: int = 100000      # buffer size for gathered data
    buffer_size_dataset: int = 100000       # buffer size for the provided data i.e. how much provided data to use
    gather_every: int = 1                   # how often we collect transition data
    gather_n: int = 1                       # how many transitions we collect at once
    

@dataclass
class DQNHyperparameters(BaseHyperparameters):
    gamma: float = 0.99                     # discount factor for Bellman Equation
    lr: float = 2.5e-4                      # learning rate for model weights
    eps_decay: float = 40000                # decay factor for epsilon greedy strategy
    eps_min: float = 0.01                   # min value for epsilon greedy strategy
    eps_max: float = 1.0                    # max value for epsilon greedy strategy
    model_hidden_layer_size: int = 64       # layer size for hidden layers in neural net
