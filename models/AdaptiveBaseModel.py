import torch
import itertools
from models.lut.generator_3dlut import Generator_3DLUT_identity, Generator_3DLUT_supplement
from models.lut.total_variation import TV_3D
from models.classifier.build import build_classifier
import logging
from engine.log.logger import setup_logger
import engine.comm as comm
from engine.checkpoint.functional import get_model_state_dict, load_model_state_dict, load_checkpoint_state_dict
from engine.slover.lr_scheduler import build_lr_scheduler


class AdaptiveBaseModel:
    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        self.device = cfg.MODEL.DEVICE

        self.lut0 = Generator_3DLUT_identity(cfg.MODEL.LUT.DIMS, cfg.MODEL.DEVICE)
        self.lut1 = Generator_3DLUT_supplement(cfg.MODEL.LUT.DIMS, cfg.MODEL.LUT.SUPPLEMENT_NUMS, cfg.MODEL.DEVICE, cfg.MODEL.LUT.ZERO_LUT)

        self.classifier = build_classifier(cfg)
        self.classifier.init_normal_classifier()

        self.tv3 = TV_3D(cfg.MODEL.LUT.DIMS, cfg.MODEL.DEVICE)

        parameters = self.lut1.parameters()
        parameters.insert(0, self.lut0.parameters())
        parameters.insert(0, self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(itertools.chain(*parameters), lr=cfg.SOLVER.BASE_LR, betas=(cfg.SOLVER.ADAM.B1, cfg.SOLVER.ADAM.B2))

        self.scheduler = build_lr_scheduler(cfg, self.optimizer_G)

        self.lambda_class_smooth = cfg.SOLVER.LAMBDA_CLASS_SMOOTH
        self.lambda_smooth = cfg.SOLVER.LAMBDA_SMOOTH
        self.lambda_monotonicity = cfg.SOLVER.LAMBDA_MONOTONICITY

        logging.getLogger(__name__).info("select {}/{} as trainer model, select zero lut ({}) form supplement".format(cfg.MODEL.ARCH, self.__class__, cfg.MODEL.LUT.ZERO_LUT))
        return

    def __call__(self, x, gt, epoch=None):
        raise RuntimeError('must be implementation __call__')

    def generator(self, img):
        pred = self.classifier(img).squeeze()

        assert pred.shape[-1] - 1 == len(self.lut1)

        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)

        shape = pred.shape[0], 1, 1, 1
        combine_a = pred[:, 0].reshape(shape) * self.lut0(img)
        lut1 = self.lut1(img)
        for i, val in lut1.items():
            combine_a += pred[:, i + 1].reshape(shape) * val

        weights_norm = torch.mean(pred ** 2)
        return combine_a, weights_norm

    @torch.no_grad()
    def generate_lut(self, img):
        pred = self.classifier(img).squeeze()

        assert pred.shape[-1] - 1 == len(self.lut1)

        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)

        combine_lut = pred[:, 0] * self.lut0.lut
        for i, lut in self.lut1.foreach():
            combine_lut += pred[:, i + 1] * lut.lut
        return torch.clip(combine_lut, 0.0, 1.0)

    def enable_train(self):
        self.lut0.train()
        self.lut1.train()
        self.classifier.train()
        return

    def disable_train(self):
        self.lut0.eval()
        self.lut1.eval()
        self.classifier.eval()
        return

    def get_state_dict(self):
        models = dict()
        models['lut'] = {0: get_model_state_dict(self.lut0)}
        models['lut'].update(self.lut1.state_dict(offset=1))
        models['cls'] = get_model_state_dict(self.classifier)
        return models

    def load_state_dict(self, state_dict: dict):
        load_model_state_dict(self.lut0, state_dict['lut'][0])
        self.lut1.load_state_dict(state_dict['lut'], offset=1)
        load_model_state_dict(self.classifier, state_dict['cls'])
        return

    def get_addition_state_dict(self):
        addition = dict()
        addition['opt_g'] = get_model_state_dict(self.optimizer_G)
        addition['scheduler'] = get_model_state_dict(self.scheduler)
        return addition

    def load_addition_state_dict(self, state_dict: dict):
        load_checkpoint_state_dict(self.optimizer_G, state_dict['opt_g'])
        scheduler = state_dict.get('scheduler', None)
        if scheduler is not None:
            load_checkpoint_state_dict(self.scheduler, scheduler)
        return

    def enable_distribute(self, cfg):
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            self.lut0 = torch.nn.parallel.DistributedDataParallel(self.lut0, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            lut1 = dict()
            for k, v in self.lut1.foreach():
                lut1[k] = torch.nn.parallel.DistributedDataParallel(v, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            self.lut1.upate(lut1)

            self.classifier = torch.nn.parallel.DistributedDataParallel(self.classifier, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(__name__).info('launch model by parallel')
            self.lut0 = torch.nn.parallel.DataParallel(self.lut0)
            lut1 = dict()
            for k, v in self.lut1.foreach():
                lut1[k] = torch.nn.parallel.DataParallel(v)
            self.lut1.upate(lut1)

            self.classifier = torch.nn.parallel.DataParallel(self.classifier)
        else:
            logging.getLogger(__name__).info('launch model by singal machine')
        return
