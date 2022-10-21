from dataclasses import dataclass

from minerl3161.hyperparameters import ClassicControlDQNHyperparameters


@dataclass
class ClassicControlRainbowDQNHyperparameters(ClassicControlDQNHyperparameters):
    # PER parameters
    prior_eps: float = 1e-6
    alpha: float = 0.2
    beta_max: float = 1.0
    beta_min: float = 0.6
    beta_final_step: int = 80_000
    # Categorical DQN parameters
    v_min: float = 0.0
    v_max: float = 200.0
    atom_size: int = 51
    # N-step Learning
    n_step: int = 3
    noisy_init: float = 0.1
    model_hidden_layer_size: int = 128  # layer size for hidden layers in neural net
