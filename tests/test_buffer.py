import os

import numpy as np
from minerl3161.buffer import ReplayBuffer, Transition
from minerl3161.utils import sample_np_state


def test_create_buffer():
    n = 10
    state_space = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }

    buffer = ReplayBuffer(n=n, obs_space=state_space)

    state = sample_np_state(state_space, features=state_space.keys())
    action = np.ones(1)
    next_state = sample_np_state(state_space, features=state_space.keys())
    reward = 1
    done = False

    transition = Transition(state, action, next_state, reward, done)

    buffer.add(*transition)
    buffer.add(state, action, next_state, reward, done)

    s, a, s_, r, d = buffer[0]

    for feature in state_space.keys():
        assert (s[feature] == state[feature]).all()
        assert (s_[feature] == next_state[feature]).all()
    assert (a == action).all()
    assert r == reward
    assert d == done

    assert len(buffer) == 2


def test_buffer_save_load():
    n = 10
    state_space = {
        "pov": np.zeros((3, 64, 64)),
        "f2": np.zeros(4),
        "f3": np.zeros(6),
    }

    save_path = "test_buffer.pickle"

    buffer = ReplayBuffer(n=n, obs_space=state_space)

    state = sample_np_state(state_space, features=state_space.keys())
    action = np.ones(1)
    next_state = sample_np_state(state_space, features=state_space.keys())
    reward = 1
    done = False

    buffer.add(state, action, next_state, reward, done)
    buffer.add(state, action, next_state, reward, done)
    buffer.add(state, action, next_state, reward, done)

    buffer.save(save_path=save_path)

    new_buffer = ReplayBuffer.load(save_path)

    # buffer and new_buffer should now be the same

    assert len(buffer) == len(new_buffer)

    s, *_ = buffer[0]
    s_, *_ = new_buffer[0]

    for feature in state_space.keys():
        assert (s[feature] == s_[feature]).all()

    os.remove(save_path)
