from dataclasses import dataclass, field
from typing import Union, List

from minerl3161.hyperparameters import BaseHyperparameters
from minerl3161.utils import linear_sampling_strategy


@dataclass
class DQNHyperparameters(BaseHyperparameters):
    gamma: float = 0.99  # discount factor for Bellman Equation
    lr: float = 2.5e-4  # learning rate for model weights
    eps_decay: float = 300_000  # decay factor for epsilon greedy strategy
    eps_min: float = 0.02  # min value for epsilon greedy strategy
    eps_max: float = 1.0  # max value for epsilon greedy strategy
    model_hidden_layer_size: int = 128  # layer size for hidden layers in neural net
    hard_update_freq: Union[
        int, None
    ] = 30_000  # how ofter to do a hard copy from q1 to q2
    soft_update_freq: Union[
        int, None
    ] = 0  # how often to do a soft update from q1 to q2
    polyak_tau: float = 1e-5  # controls the weight of the soft update
    reward_scale: float = 1.0  # controls whether we want to scale the rewards in the loss function
    
    # these are the feature names that are passed into the model to learn on
    feature_names: List = field(default_factory=lambda:[
        "pov",
        # "inventory",
        "compass",
    ])

    # these are the feature names that are passed into the observation wrapper.
    inventory_feature_names: List = field(default_factory=lambda: [
        # either set "all" or individually specify items
        "all",
    ])
    mlp_output_size: int = 64
    sampling_strategy: callable = linear_sampling_strategy