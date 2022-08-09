import os
from datetime import datetime
from tkinter import ALL

import minerl
import gym
import numpy as np
import tqdm
from minerl.data import BufferedBatchIter
from sklearn.cluster import KMeans

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
ENV_STRINGS = ['Treechop', 'Navigate', 'NavigateDense', 'NavigateExtreme', 'NavigateExtremeDense', 
                'ObtainDiamond', 'ObtainDiamondDense', 'ObtainIronPickaxe', 'ObtainIronPickaxeDense']

StringBuilder = lambda ENV_STRING: (f'MineRL{ENV_STRING}VectorObf-v0', f'src/actions/actions-{ENV_STRING}.npy')

NUM_CLUSTERS = 12 # Number of Macro Actions we want to extract

CHAIN_LEN = 10
EP_COUNT = 10 
NUM_BATCHES = 1000
MAX_ACTIONS = 100000
NUM_EPOCHS = 2
BATCH_SIZE = 10
ACTION_SIZE = 64


# Initial setup
data_path = os.path.join(os.getcwd(), "data")

if not os.path.exists(data_path):
    os.mkdir(data_path)

os.environ['MINERL_DATA_ROOT'] = data_path # Important


def extract_n_clusters(num_clusters, data):
    # Load the dataset storing NUM_BATCHES batches of actions
    act_vectors = []
    for _, act, _, _,_ in tqdm.tqdm(data.batch_iter(batch_size=BATCH_SIZE, seq_len=CHAIN_LEN, num_epochs=NUM_EPOCHS, preload_buffer_size=20)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > NUM_BATCHES:
            break

    # Reshape these the action batches
    acts = np.concatenate(act_vectors).reshape(-1, ACTION_SIZE) 
    kmeans_acts = acts[:MAX_ACTIONS]

    # Use sklearn to cluster the demonstrated actions
    return KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_acts)


def download_data():
    for ENV_STRING in ENV_STRINGS:
        ENVIRONMENT, _ = StringBuilder(ENV_STRING)

        # Downloading environment data if it doesn't exist
        env_data_path = os.path.join(data_path, ENVIRONMENT)
        if not os.path.exists(env_data_path):
            os.mkdir(f'data/{ENVIRONMENT}')
            minerl.data.download(data_path, environment = ENVIRONMENT)


def extract_actions():
    ALL_ACTIONS = []

    for ENV_STRING in ENV_STRINGS:
        ENVIRONMENT, _ = StringBuilder(ENV_STRING)

        data = minerl.data.make(environment = ENVIRONMENT)

        kmeans = extract_n_clusters(NUM_CLUSTERS, data)

        ALL_ACTIONS.append(kmeans.cluster_centers_)

    print(len(np.unique(ALL_ACTIONS)))

    np.save("src/actions/all-actions.npy", kmeans.cluster_centers_)


download_data()
extract_actions()