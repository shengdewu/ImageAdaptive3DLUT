import torch.distributed as dist
import torch
import torch.multiprocessing as mp
from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import engine.comm as comm
from fvcore.common.file_io import PathManager
from engine.log.logger import setup_logger
import logging
from abc import ABC
import abc


class BaseScheduler(ABC):
    def __init__(self):
        return

    @abc.abstractmethod
    def lunch_func(self, cfg, args):
        """
        create trainer, load traienr, start train

        Args:
            cfg :ymal
            args :cmd
        """
        pass

    def main_func(self, args, local_rank=None, global_rank=0, world_size=1, is_distributed=False):
        cfg = self.setup(args)

        cfg.defrost()
        if is_distributed:
            cfg.MODEL.TRAINER.TYPE = 1 # -1 :normal, 0:parallel, 1 :distributed
            cfg.MODEL.TRAINER.GPU_ID = local_rank
            cfg.MODEL.TRAINER.GLOBAL_RANK = global_rank
            cfg.MODEL.TRAINER.WORLD_SIZE = world_size
            cfg.MODEL.TRAINER.NUM_PER_GPUS = args.num_gpus
            cfg.MODEL.DEVICE = 'cuda'
        else:
            if torch.cuda.is_available() and args.num_gpus >= 2:
                cfg.MODEL.TRAINER.TYPE = 0  # -1 :normal, 0:parallel, 1 :distributed
                cfg.MODEL.TRAINER.GPU_ID = None
                cfg.MODEL.TRAINER.GLOBAL_RANK = 0
                cfg.MODEL.TRAINER.WORLD_SIZE = 1
                cfg.MODEL.TRAINER.NUM_PER_GPUS = args.num_gpus
                cfg.MODEL.DEVICE = 'cuda'
            else:
                cfg.MODEL.TRAINER.TYPE = -1  # -1 :normal, 0:parallel, 1 :distributed
                cfg.MODEL.TRAINER.GPU_ID = None
                cfg.MODEL.TRAINER.GLOBAL_RANK = 0
                cfg.MODEL.TRAINER.WORLD_SIZE = 1
                cfg.MODEL.TRAINER.NUM_PER_GPUS = 1
                cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        cfg.freeze()

        rank = comm.get_rank()  # rank of current node,[0, nodes - 1] and nodes : total number of nodes
        logger = setup_logger(cfg.OUTPUT_DIR, rank, name=cfg.OUTPUT_LOG_NAME)
        if comm.is_main_process() and cfg.OUTPUT_DIR:
            logger.info("is distribute {} Running with full config:\n{}".format(is_distributed, cfg.dump()))

        self.lunch_func(cfg, args)
        return

    def distributed(self, local_rank, loop_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args):
        assert torch.cuda.is_available(), "cuda is not available. Please check your installation."

        global_rank = machine_rank * num_gpus_per_machine + local_rank

        torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=global_rank)

        # Setup the local process group (which contains ranks within the same machine)
        assert comm._LOCAL_PROCESS_GROUP is None
        num_machines = world_size // num_gpus_per_machine
        for i in range(num_machines):
            ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
            pg = dist.new_group(ranks_on_i)
            if i == machine_rank:
                comm._LOCAL_PROCESS_GROUP = pg

        assert num_gpus_per_machine <= torch.cuda.device_count()

        torch.cuda.set_device(local_rank)

        comm.synchronize()

        loop_func(args, local_rank, global_rank, world_size, True)

        return

    @staticmethod
    def _find_free_port():
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Binding to port 0 will cause the OS to find an available port for us
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        # NOTE: there is still a chance the port could be taken by other processes.
        return port

    def setup(self, args):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        if comm.is_main_process() and cfg.OUTPUT_DIR:
            PathManager.mkdirs(cfg.OUTPUT_DIR)
        return cfg

    def schedule(self):
        args = default_argument_parser().parse_args()

        num_gpus_per_machine = args.num_gpus
        num_machines = args.num_machines
        machine_rank = args.machine_rank  #current machine node no.
        dist_url = args.dist_url

        world_size = num_machines * num_gpus_per_machine
        if args.distribute and world_size > 1:
            # https://github.com/pytorch/pytorch/pull/14391
            # TODO prctl in spawned processes

            if dist_url == "auto":
                assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
                port = self._find_free_port()
                dist_url = f"tcp://127.0.0.1:{port}"
            if num_machines > 1 and dist_url.startswith("file://"):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
                )

            mp.spawn(self.distributed, nprocs=num_gpus_per_machine, args=(self.main_func,
                                                                          world_size,
                                                                          num_gpus_per_machine,
                                                                          machine_rank,
                                                                          dist_url,
                                                                          args))
        else:
            self.main_func(args, is_distributed=False)

        return

