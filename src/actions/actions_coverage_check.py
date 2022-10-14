import gym
import minerl
import os
import numpy as np
import pickle
from tqdm import tqdm
from minerl.data import BufferedBatchIter

import minerl3161


def download_data(data_path: str, ENV_STRING: str):
    # Downloading environment data if it doesn't exist
    env_data_path = os.path.join(data_path, f'MineRL{ENV_STRING}-v0')
    if not os.path.exists(env_data_path):
        os.mkdir(f'data/MineRL{ENV_STRING}-v0')
        minerl.data.download(data_path, environment=f'MineRL{ENV_STRING}-v0')


def get_all_possible_action_enums():
    data_path = os.path.join(os.getcwd(), "data")

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    os.environ['MINERL_DATA_ROOT'] = data_path # Important

    ENV_STRINGS = ["ObtainDiamond"]      # , "Navigate", "NavigateExtreme", "ObtainIronPickaxe", "ObtainDiamond"
    actions = []

    for ENV_STRING in ENV_STRINGS:
        download_data(data_path, ENV_STRING)
    
    for ENV_STRING in ENV_STRINGS:
        data = minerl.data.make(environment=f'MineRL{ENV_STRING}-v0')
        iterator = BufferedBatchIter(data)

        for _, action, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
            actions.append(action)
    
    return actions


def get_extracted_actions(file_name: str = "all-actions.pickle"):
    filepath = os.path.join(minerl3161.actions_path, file_name)

    with open(filepath, 'rb') as f:
            actions = pickle.load(f)

    return actions


def create_coverage_dict(base: bool = False):
    coverage_dict = {}
    actions = get_all_possible_action_enums() if base else get_extracted_actions() 

    for action in actions:
        for key, value in action.items():
            if key == "camera":
                continue
            if key not in coverage_dict.keys():
                coverage_dict[key] = []
            value = value.tolist() if type(value) == np.ndarray else value
            if value not in coverage_dict[key]:
                coverage_dict[key].append(value)
    
    return coverage_dict


for k, v in create_coverage_dict(base = True).items(): print(f"{str.upper(k)}: {v}") if k != "camera" else print("")
print("-"*30)
# for k, v in create_coverage_dict().items(): print(f"{str.upper(k)}: {v}") if k != "camera" else print("")