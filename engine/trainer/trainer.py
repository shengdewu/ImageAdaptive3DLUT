import abc


class BaseTrainer(abc.ABC):
    """
    first:
        1. create model in __init__
        2. create dataloader in __init__
        3. create checkpointer in __init__
        4. init log use setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__) in engine.log.logger
        4. training in loop
        5. resume model in resume_or_load
    then:
        called by subclass of BaseScheduler in engine/schedule.py that must be implement abc lunch_func
    eg:
        def lunch_func(self, cfg, args):
            trainer = BaseTrainer(cfg)
            trainer.resume_or_load(args.resume)
            trainer.loop()
        return
    ps:
        model may be by registered use Registry (from fvcore.common.registry import)
    """
    def __init__(self):
        """
        eg:
            self.model = build_model(cfg)
            self.model.enable_train()
        """
        return

    @abc.abstractmethod
    def loop(self):
        raise NotImplemented('the loop must be implement')

    @abc.abstractmethod
    def resume_or_load(self, resume=False):
        """
        :param resume:
        :return:

        eg:
            model_state_dict, addition_state_dict, start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
            self.start_iter = start_iter
            if model_state_dict is not None:
                self.model.load_state_dict(model_state_dict)
            if addition_state_dict is not None:
                self.model.load_addition_state_dict(addition_state_dict)
            logging.getLogger(__name__).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        """
        raise NotImplemented('the loop must be implement')
