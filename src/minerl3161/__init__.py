import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, "..", "..", "data")
actions_path = os.path.join(dir_path, "..", "..", "data", "action_sets")

human_data_pkl_path = os.path.join(dir_path, "..", "..", "data", "human-xp.pkl")
evaluator_video_path = os.path.join(dir_path, "video")