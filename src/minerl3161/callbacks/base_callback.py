from typing import Dict, Any
from abc import ABC

import torch as th

class BaseCallback(ABC):
    def on_end_loop(self, t) -> Dict[str, Any]:
        return {}


class UnfreezeModelAfter(BaseCallback):
    def __init__(self, unfreeze_model: th.nn.Module, unfreeze_after) -> None:
        super().__init__()

        self.unfreeze_model = unfreeze_model
        self.unfreeze_model.requires_grad_(False)
        
        self.unfreeze_after = unfreeze_after

        self.triggered = False
    
    def on_end_loop(self, t):
        if t >= self.unfreeze_after and not self.triggered:
            self.unfreeze_model.requires_grad_(True)
            self.triggered = True
        
        return {"unfreezer_triggered": self.triggered}