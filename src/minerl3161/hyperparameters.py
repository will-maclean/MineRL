from dataclasses import dataclass, field
from typing import List, Union

from minerl3161.utils import linear_sampling_strategy


@dataclass
class BaseHyperparameters:
    train_steps: int = 2_000_000  # number of train_loop steps
    burn_in: int = 10_000  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 50_000  # how many steps per evaluation call
    evaluate_episodes: int = 1  # how many episodes we complete each evaluation call
    batch_size: int = 128  # batch size for training
    buffer_size_gathered: int = 250_000  # buffer size for gathered data
    buffer_size_dataset: int = (
        75_000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 1  # how many transitions we collect at once
    sampling_step: int = 1 # sampling strategy: batch includes one less human data item every time
    checkpoint_every: Union[int, None] = 200_000  # how often we should save a copy of the agent 

    # observation space
    n_stack: int = 4  # how many frames to stack
    resize_w: int = 20  # width of the resized frame
    resize_h: int = 20  # height of the resized frame
    img_feature_name: str = "pov"  # name of the image feature in the observation space
    include_equipped_items: bool = False  # whether to include the equipped items in the observation space
    inv_feature_max: int = 16 

    # sampling options
    sample_max: float = 1.0
    sample_min: float = 0.05
    sample_final_step: int = 1_500_000


@dataclass
class DQNHyperparameters(BaseHyperparameters):
    gamma: float = 0.99  # discount factor for Bellman Equation
    lr: float = 2.5e-4  # learning rate for model weights
    eps_decay: float = 250_000  # decay factor for epsilon greedy strategy
    eps_min: float = 0.02  # min value for epsilon greedy strategy
    eps_max: float = 1.0  # max value for epsilon greedy strategy
    model_hidden_layer_size: int = 64  # layer size for hidden layers in neural net
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


@dataclass
class RainbowDQNHyperparameters(DQNHyperparameters):
    # PER parameters
    prior_eps: float = 1e-6
    alpha: float = 0.2
    beta_max: float = 1.0
    beta_min: float = 0.6
    beta_final_step: int = 1_500_000
    # Categorical DQN parameters
    v_min: float = 0.0
    v_max: float = 200.0
    atom_size: int = 51
    # N-step Learning
    n_step: int = 3
    noisy_init: float = 0.1

@dataclass
class CartpoleDQNHyperparameters:
    train_steps: int = 50_000  # number of train_loop steps
    burn_in: int = 500  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 1_000  # how many steps per evaluation call
    evaluate_episodes: int = 5  # how many episodes we complete each evaluation call
    batch_size: int = 16  # batch size for training
    buffer_size_gathered: int = 75_000  # buffer size for gathered data
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
    ] = 0  # how ofter to do a hard copy from q1 to q2
    soft_update_freq: Union[
        int, None
    ] = 1  # how often to do a soft update from q1 to q2
    polyak_tau: float = 0.001  # controls the weight of the soft update
    reward_scale: float = 1.0  # controls whether we want to scale the rewards in the loss function
    
    # these are the feature names that are passed into the model to learn on
    feature_names: List = field(default_factory=lambda:[
        "state"
    ])

    mlp_output_size: int = 16
    sampling_strategy: callable = linear_sampling_strategy


@dataclass
class CartPoleRainbowDQNHyperparameters(CartpoleDQNHyperparameters):
    # PER parameters
    prior_eps: float = 1e-6
    alpha: float = 0.2
    beta_max: float = 1.0
    beta_min: float = 0.6
    beta_final_step: int = 50_000
    # Categorical DQN parameters
    v_min: float = 0.0
    v_max: float = 200.0
    atom_size: int = 51
    # N-step Learning
    n_step: int = 1
    noisy_init: float = 0.1