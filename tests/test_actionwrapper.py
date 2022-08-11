import gym
import minerl

from minerl3161.wrappers import MineRLDiscreteActionWrapper


def test_actionwrapper(minerl_env):

    act_wrapper = MineRLDiscreteActionWrapper(minerl_env)

    action_count = act_wrapper.get_actions_count()

    for action_idx in range(action_count):
        action = act_wrapper.get_action(action_idx)

        try: 
            _, _, done, _ = minerl_env.step(action)

            if done:
                minerl_env.reset()

        except Exception:
           raise InvalidActionException



class InvalidActionException(Exception):
    pass