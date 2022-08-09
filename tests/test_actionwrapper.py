import gym
import minerl

from minerl3161.wrappers import MineRLDiscreteActionWrapper


def test_actionwrapper_same_env():
    minerl_env = gym.make('MineRLObtainDiamondShovel-v0')
    minerl_env.reset()

    act_wrapper = MineRLDiscreteActionWrapper(minerl_env)

    action_count = act_wrapper.get_actions_count()

    for action_idx in range(action_count):
        action = act_wrapper.get_action(action_idx)

        try: 
            _, _, done, _ = minerl_env.step(action)

            if done:
                minerl_env.reset()

        except Exception:
           minerl_env.close()
           raise InvalidActionException

    minerl_env.close() 


class InvalidActionException(Exception):
    pass


test_actionwrapper_same_env()