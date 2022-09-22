from multiprocessing.sharedctypes import Value
import os
import wandb

from pathlib import Path


class Checkpointer:
    def __init__(self, agent, checkpoint_every: int = None, use_wandb: bool = True) -> None:
        """Checkpointer class manager agent checkpointing. Currently only checkpointing with wandb is supported.

        Args:
            agent (BaseAgent): The agent to checkpoint
            checkpoint_every (int, optional): checkpoint every n timesteps. Defaults to None.
            use_wandb (bool, optional): whether or not to checkpoint with wandb. Defaults to True.

        Raises:
            ValueError: Throws error if checkpoint_every is set, but use_wandb == False.
        """
        self.agent = agent
        
        self.checkpoint_every = checkpoint_every
        self.use_wandb = use_wandb

        if checkpoint_every is not None:
            self.active = True
        else:
            self.active = False
        
        if self.active and not use_wandb:
            print("Currently, checkpointing only supports checkpointing with wandb")
            self.active = False
    
    def step(self, timestep: int) -> dict:
        """Decides whether or not checkpoint. Called every timestep

        Args:
            timestep (int): the current timestep

        Returns:
            dict: any information to log
        """
        log_dict = {}
        log_dict["checkpointed"] = False

        if not self.active:
            return log_dict
        
        if timestep % self.checkpoint_every == 0:
            pth = os.path.join(wandb.run.dir, "checkpoints")
            Path(pth).mkdir(parents=True, exist_ok=True)
            pth = os.path.join(pth, f"ckpt_{timestep}.zip")
            self.make_checkpoint(pth)

            log_dict["checkpointed"] = True
        
        return log_dict
    
    def make_checkpoint(self, pth: str) -> dict:
        """creates a checkpoint at the specified location

        Args:
            pth (str): where to make the checkpoint (full path, not just folder)

        Returns:
            dict: any information to log
        """
        self.agent.save(pth)

        return {}

