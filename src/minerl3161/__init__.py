import os 
from .trainer import *

dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, "..", "..", "data")
actions_path = os.path.join(dir_path, "..", "actions")