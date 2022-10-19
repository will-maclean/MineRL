from typing import Dict, Any
from abc import ABC

import torch as th

class BaseCallback(ABC):
    """Callbacks provide an extensible way of adding functionality to trainers without needing to modify trainer code.
    """

    def on_end_loop(self, t: int) -> Dict[str, Any]:
        """Callback called on end of each train iteration

        Args:
            t (int): current train step

        Returns:
            Dict[str, Any]: any log dictionary
        """
        return {}


class UnfreezeModelAfter(BaseCallback):
    """Controls when to unfreeze weights on a given model after a specified number of steps
    """

    def __init__(self, unfreeze_model: th.nn.Module, unfreeze_after: int) -> None:
        """Constructor

        Args:
            unfreeze_model (th.nn.Module): model to unfreeze
            unfreeze_after (int): train step to unfreeze the model
        """
        super().__init__()

        self.unfreeze_model = unfreeze_model
        self.unfreeze_model.requires_grad_(False)
        
        self.unfreeze_after = unfreeze_after

        self.triggered = False
    
    def on_end_loop(self, t: int) -> Dict[str, Any]:
        """will unfreeze the model if t > self.unfreeze_after

        Args:
            t (int): current train step

        Returns:
            Dict[str, Any]: log dictionary
        """
        if t >= self.unfreeze_after and not self.triggered:
            self.unfreeze_model.requires_grad_(True)
            self.triggered = True
        
        return {"unfreezer_triggered": self.triggered}