from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class BaseHyperparameters:
    train_steps: int = 100000  # number of train_loop steps
    burn_in: int = 50  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 10_000  # how many steps per evaluation call
    evaluate_episodes: int = 5  # how many episodes we complete each evaluation call
    batch_size: int = 8  # batch size for training
    buffer_size_gathered: int = 100000  # buffer size for gathered data
    buffer_size_dataset: int = (
        100000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 1  # how many transitions we collect at once
    sampling_step: int = 1 # sampling strategy: batch includes one less human data item every time


@dataclass
class DQNHyperparameters(BaseHyperparameters):
    gamma: float = 0.99  # discount factor for Bellman Equation
    lr: float = 2.5e-4  # learning rate for model weights
    eps_decay: float = 40000  # decay factor for epsilon greedy strategy
    eps_min: float = 0.01  # min value for epsilon greedy strategy
    eps_max: float = 1.0  # max value for epsilon greedy strategy
    model_hidden_layer_size: int = 64  # layer size for hidden layers in neural net
    hard_update_freq: Union[
        int, None
    ] = 1000  # how ofter to do a hard copy from q1 to q2
    soft_update_freq: Union[
        int, None
    ] = 1  # how often to do a soft update from q1 to q2
    polyak_tau: float = 0.01  # controls the weight of the soft update
    reward_scale: float = 1.0  # controls whether we want to scale the rewards in the loss function
    
    # these are the feature names that are passed into the model to learn on
    feature_names: List = field(default_factory=lambda:[
        "pov",
        "inventory"
    ])

    # these are the feature names that are passed into the observation wrapper.
    inventory_feature_names: List = field(default_factory=lambda: [
        # camera
        "pov",
        # either set "all" or individually specify items
        "all",
    ])
    mlp_output_size = 64
