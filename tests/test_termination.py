from minerl3161.termination import *

def test_avg_ep_ret_con():
    termination = AvgEpisodeReturnTerminationCondition(termination_avg=100, window=10)

    for _ in range(4):
        condition = termination(episode_return=100)

        assert not condition
    
    condition = termination(episode_return=1000)

    assert condition

def test_get_termination_conditions():
    known_envs = ["CartPole-v0", "CartPole-v1"]

    for ke in known_envs:
        conditions = get_termination_condition(ke)

        assert len(conditions) > 0
    
    conditions = get_termination_condition("UnknownEnv")

    assert len(conditions) == 0
