from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class BaseHyperparameters:
    train_steps: int = 100000  # number of train_loop steps
    burn_in: int = 1000  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 10_000  # how many steps per evaluation call
    evaluate_episodes: int = 5  # how many episodes we complete each evaluation call
    batch_size: int = 32  # batch size for training
    buffer_size_gathered: int = 100000  # buffer size for gathered data
    buffer_size_dataset: int = (
        100000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 1  # how many transitions we collect at once


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
    
    # these are the feature names that are passed into the model to learn on
    feature_names = [
        "pov",
        "inventory"
    ]

    # these are the feature names that are passed into the observation wrapper.
    inventory_feature_names: List = field(default_factory=lambda: [
        # camera
        "pov",
        ## RAW MATERIALS
        # woods
        "acacia_wood",
        "birch_wood",
        "dark_oak_wood",
        "jungle_wood",
        "oak_wood",
        "spruce_wood",
        "stripped_acacia_wood",
        "stripped_birch_wood",
        "stripped_dark_oak_wood",
        "stripped_jungle_wood",
        "stripped_oak_wood",
        "stripped_spruce_wood",
        # planks
        "acacia_planks",
        "birch_planks",
        "crimson_planks",
        "dark_oak_planks",
        "jungle_planks",
        "oak_planks",
        "spruce_planks",
        "warped_planks",
        # stones
        "cobblestone",
        "mossy_cobblestone",
        "smooth_stone",
        "stone",
        # coals
        "charcoal",
        "coal",
        # ores
        "diamond_ore",
        "gold_ore",
        "iron_ore",
        # processed ores
        "iron_bars",
        "gold_ingot",
        "diamond",
        ## TOOLS
        "crafting_table",
        "furnace",
        # picks
        "diamond_pickaxe",
        "golden_pickaxe",
        "iron_pickaxe",
        "netherite_pickaxe",
        "stone_pickaxe",
        "wooden_pickaxe",
        # swords
        "diamond_sword",
        "golden_sword",
        "iron_sword",
        "netherite_sword",
        "stone_sword",
        "wooden_sword",
        # shovels
        "diamond_shovel",
        "golden_shovel",
        "iron_shovel",
        "netherite_shovel",
        "stone_shovel",
        "wooden_shovel",
        ## CONSUMABLES
        "torch",
        # foods
        "beef",
        "chicken",
        "cod",
        "mutton",
        "porkchop",
        "rabbit",
        "salmon",
        "cooked_beef",
        "cooked_chicken",
        "cooked_cod",
        "cooked_mutton",
        "cooked_porkchop",
        "cooked_rabbit",
        "cooked_salmon",
        "beetroot_soup",
        "mushroom_stew",
        "rabbit_stew",
        "bread",
        ## MISC
        "shield",
        "wheat",
        "hay_block",
        "sugar",
        "sugar_cane",
    ])
    mlp_output_size = 64
