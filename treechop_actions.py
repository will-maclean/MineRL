import os
from datetime import datetime

import minerl
import gym
import numpy as np
import tqdm
from minerl.data import BufferedBatchIter
from sklearn.cluster import KMeans


OBF_ENVS = ['MineRLTreechopVectorObf-v0', "MineRLObtainDiamondVectorObf-v0"] # Options for user
ENVIRONMENT = 'MineRLTreechopVectorObf-v0'

NUM_CLUSTERS = 12 # Number of Macro Actions we want to extract
CHAIN_LEN = 10
EP_COUNT = 10 

NUM_BATCHES = 1000
MAX_ACTIONS = 100000
NUM_EPOCHS = 2
BATCH_SIZE = 10
ACTION_SIZE = 64


def extract_n_clusters(num_clusters, data):
    # Load the dataset storing NUM_BATCHES batches of actions
    act_vectors = []
    for _, act, _, _,_ in tqdm.tqdm(data.batch_iter(batch_size=BATCH_SIZE, seq_len=CHAIN_LEN, num_epochs=NUM_EPOCHS, preload_buffer_size=20)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > NUM_BATCHES:
            break # Are we biased to the start of the actions?

    # Reshape these the action batches
    acts = np.concatenate(act_vectors).reshape(-1, ACTION_SIZE) 
    kmeans_acts = acts[:MAX_ACTIONS]

    # Use sklearn to cluster the demonstrated actions
    return KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_acts)


data_path = os.path.join(os.getcwd(), "data")

if not os.path.exists(data_path):
    os.mkdir(data_path)

os.environ['MINERL_DATA_ROOT'] = data_path # Important


# Downloading environment data if not exists
env_data_path = os.path.join(data_path, ENVIRONMENT)
if not os.path.exists(env_data_path):
    minerl.data.download(data_path, environment = ENVIRONMENT) # Careful


data = minerl.data.make(environment = ENVIRONMENT)

kmeans = extract_n_clusters(NUM_CLUSTERS, data)
np.save("src/actions/actions.npy", kmeans.cluster_centers_)

# i, net_reward, done, env = 0, 0, False, gym.make('MineRLTreechopVectorObf-v0')

# for cluster_count in range(11, NUM_CLUSTERS+1):
#     kmeans = extract_n_clusters(cluster_count, data)

#     episode_rews = []

#     for ep in range(EP_COUNT):

#         obs = env.reset()
#         net_reward = 0

#         while not done:
#             # Let's use a frame skip of 4 (could you do better than a hard-coded frame skip?)
#             if i % 4 == 0:
#                 action = {
#                     'vector': kmeans.cluster_centers_[np.random.choice(cluster_count)]
#                 }

#             obs, reward, done, info = env.step(action)
#             # env.render()

#             # if reward > 0:
#                 # print("+{} reward!".format(reward))
#             net_reward += reward
#             i += 1
        
#         episode_rews.append(net_reward)

#     print(f"AVG ep reward for random sampling with {cluster_count} actions: ", sum(episode_rews)/len(episode_rews))
