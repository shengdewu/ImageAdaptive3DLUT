from engine.checkpoint.CheckpointerStateDict import CheckpointerStateDict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from fvcore.common.file_io import PathManager
from engine.log.logger import setup_logger
import engine.comm as comm
from collections import OrderedDict


class CheckpointerManager:
    def __init__(self,
                 max_iter: Optional[int] = None,
                 save_dir: str = '',
                 check_period: int = 0,
                 file_prefix: str = 'model',
                 max_keep: Optional[int] = None,
                 *, save_to_disk: bool = True):

        setup_logger(output=save_dir, distributed_rank=comm.get_rank(), name=__name__)

        self.checkpointer = CheckpointerStateDict(save_dir,
                                                  save_to_disk=save_to_disk,
                                                  )
        self.check_period = check_period
        self.max_iter = max_iter
        self.file_prefix = file_prefix
        self.max_keep = max_keep
        self.recent_checkpoints = list()
        return

    @staticmethod
    def compose_state_dict(model, iteration):
        checkpointables = model.get_addition_state_dict()
        state_dict = model.get_state_dict()
        additional_state = {"iteration": iteration}
        checkpointables.update(additional_state)
        return state_dict, checkpointables

    def save(self, model, iteration):
        iteration = int(iteration)

        if (iteration+1) % self.check_period == 0:
            state_dict, checkpointables = CheckpointerManager.compose_state_dict(model, iteration)

            name = "{}_{:07d}".format(self.file_prefix, iteration)
            self.checkpointer.save(name, state_dict, **checkpointables)
            if self.max_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if PathManager.exists(file_to_delete) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        PathManager.rm(file_to_delete)

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                state_dict, checkpointables = CheckpointerManager.compose_state_dict(model, iteration)
                name = "{}_final".format(self.file_prefix, iteration)
                self.checkpointer.save(name, state_dict, **checkpointables)
        return

    def resume_or_load(self, model_path, resume=True):
        model_state_dict, checkpointables = self.checkpointer.resume_or_load(model_path, resume=resume)
        start_iter = 0
        if checkpointables is not None and resume and self.checkpointer.has_checkpoint():
            start_iter = checkpointables.get("iteration", -1) + 1
            checkpointables.pop("iteration")
        return model_state_dict, checkpointables, start_iter

