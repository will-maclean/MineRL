from minerl3161.wrappers import minerlWrapper, MineRLWrapper



def test_actionwrapper(minerl_env):
    wrapped_env = minerlWrapper(minerl_env)
    action_count = wrapped_env.action_space.n
    
    for action_idx in range(action_count):
        _, _, done, _ = wrapped_env.step(action_idx)

        if done:
            wrapped_env.reset()


class InvalidActionException(Exception):
    pass