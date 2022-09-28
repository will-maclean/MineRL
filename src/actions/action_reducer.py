"""Script that reduces an action set's similar actions"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from actions.action_extractor import DATA_PATH

from actions_utils import decode_action, encode_action, log

# Hyperparameters
CAMERA_SHRINK = 50
SIMILARITY_THRESHOLD = 0.05

# Options
N_GENERAL_ACTIONS = 15
ACTION_SET_NAME = "all-actions.pickle"
OUT_NAME = "action-set.pickle"
CAT_VARS = ["craft", "equip", "nearbyCraft", "nearbySmelt", "place"]

# Paths
ROOT_PATH = Path(__file__).absolute().parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')
DATA_PATH = ROOT_PATH.joinpath('data')
ACTIONS_PATH = DATA_PATH.joinpath('action_sets')

# Local Functions
def load_actions():
    log("Loading Actions...")
    with open(ACTIONS_PATH.joinpath(ACTION_SET_NAME), 'rb') as f:
        d = pickle.load(f)

    og = pd.DataFrame([decode_action(a, CAMERA_SHRINK) for a in d])
    log("Actions Loaded", "SUCCESS")
    return og

def calc_distances(df):
    proc = df.drop(CAT_VARS, axis=1).loc[:N_GENERAL_ACTIONS-1]

    # Calculate Distances
    distances = []
    for arr1 in proc.values:
        dist = []
        for arr2 in proc.values:
            dist.append(np.linalg.norm(arr1 - arr2))
        distances.append(dist)

    distances = np.array(distances)
    distances = distances / np.linalg.norm(distances, axis=1)
    return distances

def calc_remove(distances):
    log("Determining Similar Actions...")
    remove = set()
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            if distances[i][j] < SIMILARITY_THRESHOLD:
                remove.add(i)
                log(f"{(i, j)} are similar")

    return remove

if __name__ == "__main__":
    all_actions = load_actions()
    distances = calc_distances(all_actions)
    remove_i = calc_remove(distances)

    log(f"Removing {len(remove_i)} Actions: {remove_i}")
    new_actions = all_actions.drop(remove_i).reset_index(drop=True)
    log("Removed", "SUCCESS")

    actions_out = [encode_action(act, camera_shrink=CAMERA_SHRINK) for _, act in new_actions.iterrows()]
    log(f"Total Actions: {len(actions_out)}")

    log("Saving actions...")
    out_path = ACTIONS_PATH.joinpath(OUT_NAME)
    with open(out_path, 'wb') as f:
        pickle.dump(actions_out, f)
    log(f"saved at {out_path}", level="SUCCESS")
