from dataclasses import dataclass

from minerl3161.hyperparameters import MineRLDQNHyperparameters


@dataclass
class MineRLRainbowDQNHyperparameters(MineRLDQNHyperparameters):
    # PER parameters
    prior_eps: float = 1e-6
    alpha: float = 0.2
    beta_max: float = 1.0
    beta_min: float = 0.6
    beta_final_step: int = 1_500_000
    # Categorical DQN parameters
    v_min: float = -200.0
    v_max: float = 200.0
    atom_size: int = 51
    # N-step Learning
    n_step: int = 5
    noisy_init: float = 0.1
    # EPS
    use_eps: bool = False