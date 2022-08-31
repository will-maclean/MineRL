import os
import pickle

import minerl3161


def load_actions(env: str) -> list:
    filepath = os.path.join(minerl3161.actions_path, f"actions-{env}.pickle")

    with open(filepath, "rb") as f:
        return pickle.load(f)


def actions_formatter(env: str) -> None:
    actions = load_actions(env)

    for i, action in enumerate(actions):
        print("-"*15)
        print(f"Action {i+1}")
        for k, v in action.items():
            if k == "camera":
                print(v)
            elif v != 0 and v != "none":
                print(str.upper(k))


actions_formatter("Navigate")