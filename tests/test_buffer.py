import os

import numpy as np
from minerl3161.buffer import ReplayBuffer, Transition


def test_create_buffer():
    n = 10
    state_shape = (3, 4)

    buffer = ReplayBuffer(n=n, state_shape=state_shape)

    state = np.ones((3, 4))
    action = np.ones(1)
    next_state = np.ones((3, 4))
    reward = 1
    done = False

    transition = Transition(state, action, next_state, reward, done)

    buffer.add(*transition)
    buffer.add(state, action, next_state, reward, done)

    s, a, s_, r, d = buffer[0]

    assert (s == state).all()
    assert (a == action).all()
    assert (s_ == next_state).all()
    assert r == reward
    assert d == done

    assert len(buffer) == 2


def test_buffer_save_load():
    n = 10
    state_shape = (3, 4)

    save_path = "test_buffer.pickle"

    buffer = ReplayBuffer(n=n, state_shape=state_shape)

    state = np.ones((3, 4))
    action = np.ones(1)
    next_state = np.ones((3, 4))
    reward = 1
    done = False

    buffer.add(state, action, next_state, reward, done)
    buffer.add(state, action, next_state, reward, done)
    buffer.add(state, action, next_state, reward, done)

    buffer.save(save_path=save_path)

    new_buffer = ReplayBuffer.load(save_path)

    # buffer and new_buffer should now be the same

    assert len(buffer) == len(new_buffer)

    s, a, *_ = buffer[0]
    s_, a_, *_ = new_buffer[0]

    assert (s == s_).all()
    assert (a == a_).all()

    os.remove(save_path)
