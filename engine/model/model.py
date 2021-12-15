import abc


class BaseModel(abc.ABC):
    """
    1. Must be Create model and optimizer in constructer
    2. loss.backward() in run_step
    3. return dict of loss item in run_step
    4. Provide inference interface named generator
    5. provide disable/enable train function named enable_train and disable_train
    6. Provide model status acquisition and setting interface named get_state_dict and load_state_dict
    7. Provide optimizer status acquisition and setting interface named get_addition_state_dict and load_addition_state_dict
    8. may be create scheduler in constructer
    9. run_step and schedule in __call__
    """
    def __init__(self):
        """
        eg:
            from engine.slover import build_optimizer_with_gradient_clipping, build_lr_scheduler

            self.model = ResNet()
            self.optimizer = build_optimizer_with_gradient_clipping(cfg, torch.optim.SGD)(
                self.model.parameters(),
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )
            self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        """
        return

    @abc.abstractmethod
    def __call__(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        eg:
                loss_dict = self.run_step(data, **kwargs)
                self.scheduler.step(epoch)
        """
        raise NotImplemented('the __call__ must be implement')

    @abc.abstractmethod
    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        raise NotImplemented('the run_step must be implement')

    @abc.abstractmethod
    def generator(self, data):
        """
        :param data: type is dict
        :return:
        """
        raise NotImplemented('the generator must be implement')

    @abc.abstractmethod
    def enable_train(self):
        raise NotImplemented('the enable_train must be implement')

    @abc.abstractmethod
    def disable_train(self):
        raise NotImplemented('the disable_train must be implement')

    @abc.abstractmethod
    def get_state_dict(self):
        raise NotImplemented('the get_state_dict must be implement')

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        raise NotImplemented('the load_state_dict must be implement')

    @abc.abstractmethod
    def get_addition_state_dict(self):
        raise NotImplemented('the get_addition_state_dict must be implement')

    @abc.abstractmethod
    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        raise NotImplemented('the load_addition_state_dict must be implement')

    @abc.abstractmethod
    def enable_distribute(self, cfg):
        """
        :param cfg:
        :return:

        eg:
            if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
                logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            elif cfg.MODEL.TRAINER.TYPE == 0:
                logging.getLogger(__name__).info('launch model by parallel')
                model = torch.nn.parallel.DataParallel(model)
            else:
                logging.getLogger(__name__).info('launch model by singal machine')

        """
        raise NotImplemented('the enable_distribute must be implement')
