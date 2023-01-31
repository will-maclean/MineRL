from dataclasses import dataclass
from typing import Union


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
        150_000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 2  # how many transitions we collect at once
    sampling_step: int = 1 # sampling strategy: batch includes one less human data item every time
    checkpoint_every: Union[int, None] = 200_000  # how often we should save a copy of the agent 

    # observation space
    n_stack: int = 4  # how many frames to stack
    resize_w: int = 16  # width of the resized frame
    resize_h: int = 16  # height of the resized frame
    img_feature_name: str = "pov"  # name of the image feature in the observation space
    include_equipped_items: bool = False  # whether to include the equipped items in the observation space
    inv_feature_max: int = 16 

    # sampling options
    sample_max: float = 1.0
    sample_min: float = 0.05
    sample_final_step: int = 1_500_000