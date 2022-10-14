from typing import Dict
import numpy as np
from torch import nn
import torch as th


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


def np_dict_to_pt(
    np_dict: Dict[str, np.ndarray], device: str = "cpu", unsqueeze=False
) -> Dict[str, th.Tensor]:
    """Convertes a dictionary of numpy arrays to a dictionary of pytorch tensors

    Args:
        np_dict (Dict[str, np.ndarray]): dictionary of np arrays
        device (str, optional): which PyTorch device to store the tensors on. Defaults to "cpu".

    Returns:
        Dict[str, th.Tensor]: dictionary of converted data
    """
    out = {}

    for key in np_dict:
        if unsqueeze:
            out[key] = th.from_numpy(np_dict[key]).unsqueeze(0).to(device, dtype=th.float32)
        else:
            out[key] = th.from_numpy(np_dict[key]).to(device, dtype=th.float32)

    return out


def pt_dict_to_np(pt_dict: Dict[str, th.Tensor]) -> Dict[str, np.ndarray]:
    """Convertes a dictionary of torch tensors to a dictionary of np arrays

    Args:
        pt_dict (Dict[str, th.Tensor]): dictionary of pytorch tensors

    Returns:
        Dict[str, np.ndarray]: dictionary of converted data
    """
    out = {}

    for key in pt_dict:
        out[key] = pt_dict[key].detach().cpu().numpy()

    return out


def sample_pt_state(observation_space, features, device="cpu", batch=None):
    state = {}
    for feature in features:
        try:
            if batch is None:
                state[feature] = th.rand(observation_space[feature].shape, device=device)
            else:
                state[feature] = th.rand(
                    (batch, *observation_space[feature].shape), device=device
                )
        except TypeError:
            continue

    return state


def sample_np_state(observation_space, features, batch=None):
    state = {}
    for feature in features:
        try:
            if batch is None:
                state[feature] = np.random.rand(*observation_space[feature].shape).astype(np.float32)
            else:
                state[feature] = np.random.rand(*(batch, *observation_space[feature].shape)).astype(np.float32)
        except TypeError:
            continue

    return state


def linear_sampling_strategy(batch_size, step, *args, **kwargs):
    """_summary_

    Args:
        dataset_size (_type_): _description_
        gathered_size (_type_): _description_
        step (int): current timestep

    Returns:
        _type_: _description_
    """
    r = linear_decay(step, *args, **kwargs)

    human_r = int(np.rint(batch_size * r).item())
    gathered_r = int(np.rint(batch_size * (1-r)).item())
    return human_r, gathered_r

def np_batch_to_tensor_batch(batch, device):
    batch["state"] = np_dict_to_pt(batch["state"], device=device)

    batch["next_state"] = np_dict_to_pt(batch["next_state"], device=device)

    batch["action"] = th.tensor(
        batch["action"], dtype=th.long, device=device
    )
    batch["reward"] = th.tensor(
        batch["reward"], dtype=th.float32, device=device
    )
    batch["done"] = th.tensor(
        batch["done"], dtype=th.float32, device=device
    )
    return batch
