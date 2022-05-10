from dataclasses import dataclass

@dataclass
class BaseHyperparameters:
    lr: float = 2.5e-4
    time_steps: int = 100000
    batch_size: int = 32
    buffer_size_gathered: int = 100000
    buffer_size_dataset: int = 100000
    gamma: float = 0.99
    

@dataclass
class DQNHyperparameters(BaseHyperparameters):
    eps_decay: float = 1e-5
    eps_min: float = 0.01
    eps_max: float = 1.0
    model_hidden_layer_size: int = 64
