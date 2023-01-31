from dataclasses import dataclass, field
from typing import Union, List

from minerl3161.utils.utils import linear_sampling_strategy


@dataclass
class CartpoleDQNHyperparameters:
    train_steps: int = 50_000  # number of train_loop steps
    burn_in: int = 500  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 1_000  # how many steps per evaluation call
    evaluate_episodes: int = 5  # how many episodes we complete each evaluation call
    batch_size: int = 16  # batch size for training
    buffer_size_gathered: int = 50_000  # buffer size for gathered data
    buffer_size_dataset: int = (
        50_000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 1  # how many transitions we collect at once
    sampling_step: int = 1 # sampling strategy: batch includes one less human data item every time
    checkpoint_every: Union[int, None] = 200_000  # how often we should save a copy of the agent 

    gamma: float = 0.99  # discount factor for Bellman Equation
    lr: float = 2.5e-4  # learning rate for model weights
    eps_decay: float = 30_000  # decay factor for epsilon greedy strategy
    eps_min: float = 0.01  # min value for epsilon greedy strategy
    eps_max: float = 1.0  # max value for epsilon greedy strategy
    model_hidden_layer_size: int = 16  # layer size for hidden layers in neural net
    hard_update_freq: Union[
        int, None
    ] = 5000  # how ofter to do a hard copy from q1 to q2
    soft_update_freq: Union[
        int, None
    ] = 0  # how often to do a soft update from q1 to q2
    polyak_tau: float = 0.001  # controls the weight of the soft update
    reward_scale: float = 1.0  # controls whether we want to scale the rewards in the loss function
    
    # these are the feature names that are passed into the model to learn on
    feature_names: List = field(default_factory=lambda:[
        "state"
    ])

    mlp_output_size: int = 16
    sampling_strategy: callable = linear_sampling_strategy