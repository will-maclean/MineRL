from minerl3161.pl_pretraining.pl_model import DQNPretrainer
from minerl3161.hyperparameters import DQNHyperparameters


def test_load_pretrained(minerl_env):
    hp = DQNHyperparameters()

    pl_agent = DQNPretrainer(
        obs_space=minerl_env.observation_space,
        n_actions=minerl_env.action_space.n,
        hyperparams=hp,
        gamma=0.99,
    )

    pl_agent