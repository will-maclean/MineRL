from multiprocessing.sharedctypes import Value
import pytest
import wandb

from minerl3161.utils.checkpointer import Checkpointer

wandb_entity = "minerl3161"
wandb_project = "testing"


class DummyAgent():
    def save(self, pth):
        pass


def notest_checkpointer_dummy():
    agent = DummyAgent()

    # test off
    ckptr = Checkpointer(agent=agent, checkpoint_every=None, use_wandb=False)
    ckptr.step(0)

    # test on but wandb off
    with pytest.raises(ValueError):
        ckptr = Checkpointer(agent=agent, checkpoint_every=10, use_wandb=False)
    
    # test on with wandb on
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
    )
    run_id = wandb.run.id
    ckptr = Checkpointer(agent=agent, checkpoint_every=10, use_wandb=True)
    log_dict = ckptr.step(100)
    assert log_dict["checkpointed"] == True

    # tidy up by deleting the run
    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    run.delete()
