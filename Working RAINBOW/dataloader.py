import numpy as np
import torch
import h5py

from replay_buffer import ReplayBuffer
# from imitation.data.types import Transitions


class Dataloader:
    def __init__(self):
        pass

    @staticmethod
    def save_to_h5(buffer: ReplayBuffer, file_name: str ="data", folder_name:str = "env"):
        training_data = Dataloader.convert_to_np(buffer)
        with h5py.File(f'data/{folder_name}/{file_name}.h5', 'w') as hf:
            hf.create_dataset("training_data",  data=training_data)
    
    @staticmethod
    def load_from_h5(file_name: str ="data", folder_name: str = "env", as_transition: bool = True, obs_space: int = 4):
        print(f'data/{folder_name}/{file_name}.h5')
        with h5py.File(f'data/{folder_name}/{file_name}.h5', 'r') as hf:
            data = hf["training_data"][:]

            if as_transition:
                states, acts, next_states, dones = Dataloader.format_data(data, obs_space)
                return Dataloader.wrap_as_transition(states, acts, next_states, dones)

            return Dataloader.format_data(data, obs_space)

    @staticmethod
    def convert_to_np(buffer: ReplayBuffer):
        states = buffer.obs_buf
        acts = buffer.acts_buf
        rewards = buffer.rews_buf
        next_states = buffer.next_obs_buf
        dones = buffer.done_buf

        data = []

        """
        For h5 to play nice with our data, it is being created as a one dimensional vector, in the following format:

        [s1, a1, r1, s'1, d1, ...]

        The values in the state vector are flattened such that the data array can remain one dimensional. The format becomes:

        [s1_1, s1_2, s1_3, s1_4, a1, r1, s'1_1, s'1_2, s'1_3, s'1_4, d1, ...]
        
        This format will be taken into consideration when the data is being loaded into memory.
        """
        
        for i in range(buffer.size):
            data.extend(states[i])
            data.extend((acts[i], rewards[i]))
            data.extend(next_states[i])
            data.append(dones[i])
        
        return np.array(data, dtype=np.float32)

    @staticmethod
    def format_data(data: np.array, obs_space: int):
        states = []
        acts = []
        rewards = []
        next_states = []
        dones = []

        experience_size = 3 + (2 * obs_space)

        for i, item in enumerate(data):
            index = i % experience_size

            # NOTE: Dynamically changes the indexing based on the size of the observation space. Assumes discrete action space.
            if index == 0:
                temp_states = []
                for j in range(obs_space):
                    temp_states.append(data[i+j])
                states.append(temp_states)
            elif index == obs_space:
                acts.append(item)
            elif index == obs_space+1:
                rewards.append(item)
            elif index == obs_space+2:
                temp_states = []
                for j in range(obs_space):
                    temp_states.append(data[i+j])
                next_states.append(temp_states)
            elif index == experience_size - 1:
                dones.append(True if item == 1 else False)

        return states, acts, next_states, dones
    
    @staticmethod
    def wrap_as_transition(states, acts, next_states, dones):
        return None
        # return Transitions(np.array(states), np.array(acts), np.array([{} for i in range(len(states))]), np.array(next_states), np.array(dones))

    @staticmethod
    def generate_train_test_dataset(data_files: list, folder_name: str, obs_space: int, split_percent: float = 0.8):
        states, acts, next_states, dones = Dataloader._combine_data(data_files, folder_name, obs_space=obs_space)
        
        exp_data = list(zip(states, acts, next_states, dones))

        train_size = int(split_percent * len(exp_data))
        test_size = len(exp_data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(exp_data, [train_size, test_size], generator=torch.Generator())
        
        train_dataset = list(zip(*train_dataset))
        test_dataset = list(zip(*test_dataset))

        tr_state, tr_act, tr_next_state, tr_done = train_dataset[0], train_dataset[1], train_dataset[2], train_dataset[3]
        te_state, te_act, te_next_state, te_done = test_dataset[0], test_dataset[1], test_dataset[2], test_dataset[3]

        return Dataloader.wrap_as_transition(tr_state, tr_act, tr_next_state, tr_done), Dataloader.wrap_as_transition(te_state, te_act, te_next_state, te_done)
    
    @staticmethod
    def generate_empty_datasets(state_space):
        return Dataloader.wrap_as_transition([[1 for x in range(state_space)]]*32, [1]*32, [[1 for x in range(state_space)]]*32, [False]*32), Dataloader.wrap_as_transition([[1 for x in range(state_space)]]*32, [1]*32, [[1 for x in range(state_space)]]*32, [False]*32)

    @staticmethod
    def generate_combined_dataset(data_files: list, folder_name: str, obs_space: int):
        states, acts, next_states, dones = Dataloader._combine_data(data_files, folder_name, obs_space=obs_space)
        return Dataloader.wrap_as_transition(states, acts, next_states, dones)
    
    @staticmethod
    def _combine_data(data_files: list, folder_name: str, obs_space: int):
        states, acts, next_states, dones = [], [], [], []

        for file in data_files:
            s, a, n_s, d = Dataloader.load_from_h5(file, folder_name, as_transition=False, obs_space=obs_space)
            states.extend(s)
            acts.extend(a)
            next_states.extend(n_s)
            dones.extend(d)
        
        return states, acts, next_states, dones
