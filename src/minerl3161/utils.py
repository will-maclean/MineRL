import numpy as np
from torch import nn


# TODO: write tests
def linear_decay(
    step: int, start_val: float, final_val: float, final_steps: int
) -> float:
    """linear decay function

    Args:
        step (int): current env step
        start_val (float): start value for decay
        final_val (float): final value for decay
        final_steps (int): timestep to reach final decay value

    Returns:
        float: current value of linear decay, given the inputs
    """
    fraction = min(float(step) / final_steps, 1.0)
    return start_val + fraction * (final_val - start_val)


# TODO: write tests
def epsilon_decay(step: int, start_val: float, final_val: float, decay: float) -> float:
    """exponential decay function

    Args:
        step (int): current env step
        start_val (float): start value for decay
        final_val (float): final value for decay
        decay (float): decay constant

    Returns:
        float: current value of epsilon decay, given inputs
    """
    return max(
        final_val + (start_val - final_val) * np.exp(-1 * step / decay), final_val
    )


# TODO: write tests
def copy_weights(copy_from: nn.Module, copy_to: nn.Module, polyak=None):
    """
    Copy weights from one network to another. Optionally copies with Polyak averaging.
    Parameters
    ----------
    copy_from - net to copy from
    copy_to - net to copy to
    polyak - if None, then don't do Polyak averaging (i.e. directly copy weights). If you want Polyak averaging, then
    set polyak to your tau constant (usually 0.01).
    Returns
    -------
    None
    """
    if polyak is not None:
        for target_param, param in zip(copy_to.parameters(), copy_from.parameters()):
            target_param.data.copy_(polyak * param + (1 - polyak) * target_param)
    else:
        copy_to.load_state_dict(copy_from.state_dict())
