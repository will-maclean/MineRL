from datetime import datetime
import numpy as np
import os
import minerl

# Globals
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

CAT_VARS = ["craft", "equip", "nearbyCraft", "nearbySmelt", "place"]

# Util Functions
get_env_name = lambda env_str: f'MineRL{env_str}-v0'

def log(msg, level="INFO"):
    format_dict = {
        "SUCCESS": "\U00002705 SUCCESS",
        "ERROR": "\U0000274C ERROR"
    }

    print(f"{datetime.now()} | {format_dict.get(level, level)} | {msg}")
    
def check_download(env):
    environment = get_env_name(env)
    # Downloading environment data if it doesn't exist
    env_data_path = os.path.join(os.environ['MINERL_DATA_ROOT'], environment)
    if not os.path.exists(env_data_path):
        log(f"Downloading {environment} data...")
        os.mkdir(env_data_path)
        minerl.data.download(environment = environment)
        log(f"Downloaded", level="SUCCESS")
    else:
        log(f"{environment} Exists")


def encode_action(obj, bin_prob_threshold = 0):
    """ Encodes the action dict into a format acceptable by minerl
    """
    proc = {}

    for k, v in obj.items():
        if 'camera' not in k:
            try:
                proc[k] = np.array(int(float(v) > bin_prob_threshold))
            except:
                proc[k] = v
    
    proc['camera'] = np.array([obj.get('camera0'), obj.get('camera1')], dtype=np.float32)
    return proc

def decode_batch(obj, batch_size) -> list:
    """Decodes the batch of actions into a list of actions sutiable to fit into a dataframe.
    Important for kmodes/kmeans
    """
    actions = []

    for i in range(batch_size):
        proc = NULL_ACTION.copy()
        for k in obj.keys():
            if k == "camera":
                for i, dim in enumerate(obj[k][i]):
                    proc[f"{k}{i}"] = dim
            else:
                proc[k] =  obj[k][i]
        actions.append(proc)

    return actions

def decode_action(obj: dict, camera_shrink_factor=1) -> list:
    """Decodes an action to fit into a dataframe.
    Helper function for MineRLWrapper.map_action()
    """
    proc = NULL_ACTION.copy()

    for k in obj.keys():
        if k == "camera":
            for d, dim in enumerate(obj[k]):
                proc[f"{k}{d}"] = dim/camera_shrink_factor
        else:
            proc[k] = obj[k] if not isinstance(obj[k], np.ndarray) else obj[k].tolist()
    return proc