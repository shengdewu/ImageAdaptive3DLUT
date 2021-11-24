import logging
import torch
from engine.slover import build_optimizer_with_gradient_clipping, build_lr_scheduler
from engine.checkpoint.CheckpointerManager import CheckpointerManager
import engine.comm as comm
import abc
from engine.log.logger import setup_logger


class BaseTrainer(abc.ABC):
    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        self.model = self.build_model(cfg)

        self.optimizer = self.build_optimizer(cfg)

        self.scheduler = self.build_lr_scheduler(cfg)

        self.dataloader = self.build_train_loader(cfg)

        self.enable_distribute(cfg)

        self.model.train()

        self.checkpoint = self.build_checkpointer(cfg)

        self.device = cfg.MODEL.DEVICE
        self.iter = 0
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.model_path = cfg.MODEL.WEIGHTS
        self.iter_train_loader = iter(self.dataloader)
        return

    @abc.abstractmethod
    def build_model(self, cfg):
        pass

    def build_optimizer(self, cfg):
        return build_optimizer_with_gradient_clipping(cfg, torch.optim.SGD)(
            self.model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    @abc.abstractmethod
    def build_train_loader(self, cfg):
        pass

    def build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, self.optimizer)

    def enable_distribute(self, cfg):
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(__name__).info('launch model by parallel')
            self.model = torch.nn.parallel.DataParallel(self.model)
        else:
            logging.getLogger(__name__).info('launch model by singal machine')
        return

    @abc.abstractmethod
    def build_checkpointer(self, cfg):
        pass

    def run_step(self):
        assert self.model.training
        data = next(self.iter_train_loader)

        loss_dict = self.model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict['lr'] = self.optimizer.param_groups[0]['lr']
        return metrics_dict

    def run_after(self, params):
        if self.iter % self.checkpoint.check_period == 0:
            logging.getLogger(__name__).info('trainer run step {}-{}'.format(self.iter, params))
        return

    def resume_or_load(self, resume=False):
        self.start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
        logging.getLogger(__name__).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        return

    def save_model(self, epoch):
        self.checkpoint.save(epoch)
        return

    def train(self):
        for i in range(self.start_iter, self.max_iter):
            self.iter = i
            metrics_dict = self.run_step()
            self.run_after(metrics_dict)
            self.scheduler.step(i)
            self.checkpoint.save(i)
        return

    @torch.no_grad()
    def test(self, *args, **kwargs):
        assert self.model.training == False
        return self.model(args)

