"""Script to extract general and functional actions"""
import os
import pickle
from pathlib import Path
from itertools import product

import minerl
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from minerl.data import BufferedBatchIter
from sklearn.cluster import KMeans

from actions_utils import (CAT_VARS, NULL_ACTION, check_download, decode_batch,
                           encode_action, get_env_name, log)

# Hyperparameters
BIN_PROB_THRESHOLD = 0.05
N_GENERAL_ACTIONS = 15
N_CAM_ACTIONS = 15

GENERAL_CFG = [
    ["ObtainDiamond", 100],
    ["ObtainIronPickaxe", 100],
    ["Navigate", 100],
    ["Treechop", 100],
]

FUNCTIONAL_CFG = [
    # ["ObtainDiamond", 200, True]
] 

OPPOSITES_PROBS = [
    {"forward": 0.05, "back": 0.95},
    {"right": 0.5, "left": 0.5},
    {"sprint": 0.25, "sneak": 0.75}
]

# Options
OUTPUT_NAME = "all-actions"
RANDOM_STATE = 123
NUM_EPOCHS = 2
BATCH_SIZE = 10
TIMEOUT_ACTIONS = 100000

# Paths
ROOT_PATH = Path(__file__).absolute().parent.parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')
DATA_PATH = ROOT_PATH.joinpath('data')
ACTIONS_PATH = DATA_PATH.joinpath('action_sets')

# Initial path setup
data_path = str(DATA_PATH)
if not os.path.exists(data_path):
    os.mkdir(data_path)
os.environ['MINERL_DATA_ROOT'] = data_path # Important

# Util functions
def test_functional(obj):
    for v in CAT_VARS:
        if obj[v] != 'none':
            return True, v
    return False, None

def extract_from_env(env, n, functional=False):
    environment = get_env_name(env)
    data = minerl.data.make(environment = environment)

    iterator = BufferedBatchIter(data)
    i = 0
    collected_actions = []
    cat_var_counter = {k:0 for k in CAT_VARS}

    for current_state, action, reward, next_state, done in iterator.buffered_batch_iter(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
        decoded = decode_batch(action, batch_size=BATCH_SIZE)
        
        if functional:
            for act in decoded:
                is_functional, var = test_functional(act)
                if is_functional and cat_var_counter[var] <= (n / 5):
                    collected_actions.append(act)
                    cat_var_counter[var] += 1

        else:
            collected_actions += decoded

        if len(collected_actions) >= n or i >= TIMEOUT_ACTIONS // BATCH_SIZE:
            break

    df = pd.DataFrame(collected_actions)

    return df

def extract_actions(cfg):
    return pd.concat([extract_from_env(*a) for a in cfg]).reset_index(drop=True)

def run_kmeans(df, num_clusters):
    """Running Kmeans on a dataset of actions"""
    log("Running KMeans...")
    df = df.drop(CAT_VARS, axis = 1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=RANDOM_STATE).fit(df.values)

    actions_list = ['attack', 'back', 'camera0', 'camera1', 
    'forward', 'jump', 'left',  'right', 'sneak','sprint']

    extracted_actions = []
    for cluster in kmeans.cluster_centers_:
        action = NULL_ACTION.copy()
        action.update({actions_list[i]: cluster[i] for i in range(len(cluster))})
        extracted_actions.append(encode_action(action, bin_prob_threshold=BIN_PROB_THRESHOLD))
    
    log("Extracted Actions", level="SUCCESS")
    return extracted_actions

def remove_opposites(actions: list) -> list:
    """Removes opposite actions ocurring at the same time using a sampling
    strategy in the `OPPOSITES_PROBS` dictionary.
    Args:
        actions (list): The kmeans or kprototypes cluster centres
    Returns:
        new_actions (list): The new action set once the opposites have been removed
    """
    new_actions = []
    for i in range(len(actions)):
        act = actions[i]
        new_act = act.copy()

        for opp in OPPOSITES_PROBS:
            key1 = list(opp.keys())[0]
            key2 = list(opp.keys())[1]
            if act[key1] and act[key2]:
                if np.random.random() < opp[key1]:
                    new_act[key2] = np.array(0)
                else:
                    new_act[key1] = np.array(0)
        
        new_actions.append(new_act)

    return new_actions

def generate_cam_actions():
    log("Collecting Camera Attack Actions...")
    cam0 = np.arange(-90, 90, 30)
    cam1 = np.arange(-90, 90, 30)
    acts = list(product(cam0, cam1))

    weights = np.array([1/(abs(a[0]) + abs(a[1]) + 1) for a in acts])
    weights = weights / weights.sum()
    choices = np.random.choice(np.arange(0, len(acts)), N_CAM_ACTIONS, p=weights, replace=False)

    chosen_angles = [acts[i] for i in range(len(acts)) if i in choices]

    camera_actions = []
    for angle in chosen_angles:
        act = NULL_ACTION.copy()
        act['camera0'], act['camera1'] = angle
        act['attack'] = np.array(1)
        camera_actions.append(encode_action(act))
    log("Collected Camera Attack Actions", "SUCCESS")
    return camera_actions

def load_func_actions():
    log("Loading Functional Actions...")
    path = ACTIONS_PATH.joinpath(f'functional-actions.pickle')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    log("Loaded Functional Actions", "SUCCESS")
    return d

def run_kprototypes(df):
    """Running Kprototypes on a dataset of actions"""
    mark_array = df.values
    categorical_features_idx = [4, 5, 9, 10, 11]

    kproto = KPrototypes(n_clusters=1, max_iter=200).fit(
        mark_array, categorical=categorical_features_idx)
    
    actions_list = ['attack', 'back', 'camera0', 'camera1', 
        'forward', 'jump', 'left',  'right', 'sneak','sprint', 
        'craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']

    extracted_actions = []
    for cluster in kproto.cluster_centroids_:
        action = {actions_list[i]: cluster[i] for i in range(len(cluster))}
        extracted_actions.append(encode_action(action, bin_prob_threshold=0.4))

    return extracted_actions


if __name__ == '__main__':
    # 1. Check Downloads
    for env_str in {i[0] for i in GENERAL_CFG + FUNCTIONAL_CFG}:
        check_download(env_str)

    # 2. Collect Actions
    log("Collecting General Actions...")
    general_df = extract_actions(GENERAL_CFG)
    log("Collected General Actions", "SUCCESS")

    if len(FUNCTIONAL_CFG) > 0:
        log("Collecting Functional Actions...")
        functional_df = extract_actions(FUNCTIONAL_CFG)
        log("Collected Functional Actions", "SUCCESS")
    else:
        func_actions = load_func_actions()

    cam_actions = generate_cam_actions()
    
    # 3. Perform Clustering
    log("Clustering General Actions...")
    gen_actions = remove_opposites(run_kmeans(general_df, N_GENERAL_ACTIONS))
    log("Clustered General Actions", "SUCCESS")

    if len(FUNCTIONAL_CFG) > 0:
        log("Clustering Functional Actions...")
        func_actions = []
        for v in CAT_VARS:
            gb = functional_df.groupby(v)
            func_actions += [run_kprototypes(gb.get_group(grp))[0] for grp in gb.groups if grp != 'none']
        log("Clustered Functional Actions", "SUCCESS")
        
    # 4. Save
    log("Saving actions...")
    out_path = ACTIONS_PATH.joinpath(f'{OUTPUT_NAME}.pickle')
    with open(out_path, 'wb') as f:
        pickle.dump(gen_actions + cam_actions + func_actions, f)
    log(f"saved at {out_path}", level="SUCCESS")
