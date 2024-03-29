from copy import deepcopy as dc
from pickle import dump

NULL_ACTION = {
    'attack': 0,
    'back': 0,
    'camera': [0, 0],
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

x_angles = [-30, 30]

actions = []

for angle in x_angles:
    new_action = dc(NULL_ACTION)
    new_action["camera"][1] = angle
    actions.append(new_action)

new_action = dc(NULL_ACTION)
new_action["forward"] = 1
new_action["jump"] = 1
actions.append(new_action)

new_action = dc(NULL_ACTION)
new_action["forward"] = 1
new_action["attack"] = 1
actions.append(new_action)

with open("custom-basic-nav.pkl", "wb") as f:
    dump(actions, f)
