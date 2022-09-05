import os
from datetime import datetime
from pathlib import Path
import pickle

import gym
import minerl
import numpy as np
import pandas as pd
import tqdm
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from minerl.data import BufferedBatchIter
from numpy import array, float32

"""
This file is used to extract the n most used actions from the human priors dataset for each task.

The following is all task environment strings
    - Treechop
    - Navigate
    - NavigateDense
    - NavigateExtreme
    - NavigateExtremeDense
    - ObtainDiamond
    - ObtainDiamondDense
    - ObtainIronPickaxe
    - ObtainIronPickaxeDense
"""

# Hyperparameters
RANDOM_STATE = 123
NUM_CLUSTERS = 12 # Number of Macro Actions we want to extract

NUM_EPOCHS = 2
BATCH_SIZE = 10
MAX_ACTIONS = 20000

BIN_AS_CAT = False
BIN_PROB_THRESHOLD = 0.2

NULL_ACTION = {
    'attack': 0,
    'back': 0,
    'camera0': 0.0,
    'camera1': 0.0,
    'craft': 'none',
    'equip': 'none',
    'forward': 0,
    'jump': 0,
    'left': 0,
    'nearbyCraft': 'none',
    'nearbySmelt': 'none',
    'place': 'none',
    'right': 0,
    'sneak': 0,
    'sprint': 0
}

# Paths
ROOT_PATH = Path(__file__).absolute().parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')
DATA_PATH = ROOT_PATH.joinpath('data')
ACTIONS_PATH = SRC_PATH.joinpath('actions')

# Util functions
StringBuilder = lambda ENV_STRING: (f'MineRL{ENV_STRING}-v0', str(ACTIONS_PATH.joinpath(f'actions-{ENV_STRING}.pickle')))

def log(msg, level="INFO"):
    format_dict = {
        "SUCCESS": "\U00002705 SUCCESS",
        "ERROR": "\U0000274C ERROR"
    }

    print(f"{datetime.now()} | {format_dict.get(level, level)} | {msg}")

def decode_actions(obj) -> list:
    """Decodes the batch of actions into a list of actions sutiable to fit into a dataframe.
    Important for kmodes/kmeans
    """
    actions = []

    for i in range(BATCH_SIZE):
        proc = NULL_ACTION.copy()
        for k in obj.keys():
            if k == "camera":
                for i, dim in enumerate(obj[k][i]):
                    proc[f"{k}{i}"] = dim
            else:
                proc[k] =  obj[k][i]
        actions.append(proc)

    return actions

def encode_action(obj):
    """ Encodes the action dict into a format acceptable by minerl
    """
    proc = {}

    for k, v in obj.items():
        if 'camera' not in k:
            try:
                proc[k] = array(int(float(v) > BIN_PROB_THRESHOLD)) if not BIN_AS_CAT else array(int(float(v)))
            except:
                proc[k] = v
    
    proc['camera'] = array([obj.get('camera0'), obj.get('camera1')], dtype=float32)
    return proc

def run_kprototypes(df):
    """Running Kprototypes on a dataset of actions"""
    log("Running KPrototypes...")

    mark_array = df.values
    categorical_features_idx = [col for col in df.columns if 'camera' not in col]

    kproto = KPrototypes(n_clusters=NUM_CLUSTERS, max_iter=200).fit(
        mark_array, categorical=categorical_features_idx)
    
    actions_list = ['attack', 'back', 'camera0', 'camera1', 
        'forward', 'jump', 'left',  'right', 'sneak','sprint', 
        'craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']

    extracted_actions = []
    for cluster in kproto.cluster_centroids_:
        action = {actions_list[i]: cluster[i] for i in range(len(cluster))}
        extracted_actions.append(encode_action(action))

    log("Extracted Actions", level="SUCCESS")

    return extracted_actions

def run_kmeans(df):
    """Running Kmeans on a dataset of actions"""
    log("Running KMeans...")
    df = df.drop(['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place'], axis = 1)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE).fit(df.values)

    actions_list = ['attack', 'back', 'camera0', 'camera1', 
    'forward', 'jump', 'left',  'right', 'sneak','sprint']

    extracted_actions = []
    for cluster in kmeans.cluster_centers_:
        action = NULL_ACTION.copy()
        action.update({actions_list[i]: cluster[i] for i in range(len(cluster))})
        extracted_actions.append(encode_action(action))
    
    log("Extracted Actions", level="SUCCESS")
    return extracted_actions

def extract_n_clusters(data):
    # Load the dataset storing NUM_BATCHES batches of actions
    iterator = BufferedBatchIter(data)
    i = 0
    collected_actions = []
    log("Collecting actions...")
    for current_state, action, reward, next_state, done in iterator.buffered_batch_iter(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
        collected_actions += decode_actions(action)
        
        i += 1
        if i == MAX_ACTIONS // BATCH_SIZE:
            break

    df = pd.DataFrame(collected_actions)

    log("Actions Collected", level="SUCCESS")

    return run_kprototypes(df) if BIN_AS_CAT else run_kmeans(df)

def check_download(environment):
    # Downloading environment data if it doesn't exist
    env_data_path = os.path.join(os.environ['MINERL_DATA_ROOT'], environment)
    if not os.path.exists(env_data_path):
        log(f"Downloading {environment} data...")
        os.mkdir(env_data_path)
        minerl.data.download(environment = environment)
        log(f"Downloaded", level="SUCCESS")

def save(lst, path):
    log("Saving actions...")
    with open(path, 'wb') as f:
        pickle.dump(lst, f)
    log(f"saved at {path}", level="SUCCESS")

def extract_actions(ENV_STRING):
    log(f"BEGIN ACTIONS EXTRACTION, ENV: {ENV_STRING}")
    environment, save_path = StringBuilder(ENV_STRING)
    check_download(environment)

    data = minerl.data.make(environment = environment)

    actions = extract_n_clusters(data)

    save(actions, save_path)    

def merge_actions():
    action_strings = ["Treechop", "Navigate", "ObtainIronPickaxe", "ObtainDiamond"]
    log(f"Merging actions from the following files: {'.pickle, '.join(action_strings)}")
    actions = []
    for action_string in action_strings:
        _, save_path = StringBuilder(action_string)
        
        with open(save_path, 'rb') as f:
            action_set = pickle.load(f)

        actions.extend(action_set)
    
    log(f'Merged Actions, Final action set length: {len(actions)}', level="SUCCESS")

    save(actions, str(ACTIONS_PATH.joinpath("all-actions.pickle")))

if __name__ == "__main__":
    # Initial setup
    data_path = str(DATA_PATH)

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    os.environ['MINERL_DATA_ROOT'] = data_path # Important

    extract_actions("Treechop")
    extract_actions("Navigate")
    extract_actions("ObtainIronPickaxe")
    extract_actions("ObtainDiamond")
    merge_actions()


