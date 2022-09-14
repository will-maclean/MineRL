from minerl3161.wrappers import minerlWrapper



def test_actionwrapper(minerl_env):
    wrapped_env = minerlWrapper(minerl_env)
    action_count = wrapped_env.action_space.n
    
    for action_idx in range(action_count):
        action = wrapped_env.convert_action(action_idx)

        try: 
            _, _, done, _ = wrapped_env.step(action)

            if done:
                wrapped_env.reset()

        except Exception:
            raise InvalidActionException


class InvalidActionException(Exception):
    pass