import torch
import itertools
import math
from models import MODEL_ARCH_REGISTRY
from models.lut.generator_3dlut import Generator_3DLUT_identity, Generator_3DLUT_n_zero
from models.lut.total_variation import TV_3D
from models.classifier import build_classifier
import logging
from engine.log.logger import setup_logger
import engine.comm as comm


@MODEL_ARCH_REGISTRY.register()
class AdaptivePairedModel:
    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        self.device = cfg.MODEL.DEVICE
        self.lambda_smooth = cfg.SOLVER.LAMBDA_SMOOTH
        self.lambda_monotonicity = cfg.SOLVER.LAMBDA_MONOTONICITY
        self.luts_name_prefix = 'luts_paired'
        self.cls_name_prefix = 'classifier_paired'

        self.lut0 = Generator_3DLUT_identity(cfg.MODEL.LUT.DIMS, cfg.MODEL.DEVICE)
        self.lut1 = Generator_3DLUT_n_zero(cfg.MODEL.LUT.DIMS, cfg.MODEL.LUT.SUPPLEMENT_NUMS, cfg.MODEL.DEVICE)

        self.classifier = build_classifier(cfg)
        logging.getLogger(__name__).info('select {} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH))
        self.classifier.init_normal_classifier()

        self.tv3 = TV_3D(cfg.MODEL.LUT.DIMS, cfg.MODEL.DEVICE)

        # Loss functions
        self.criterion_pixelwise = torch.nn.MSELoss()

        parameters = self.lut1.parameters()
        parameters.insert(0, self.lut0.parameters())
        parameters.insert(0, self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(itertools.chain(*parameters), lr=cfg.SOLVER.BASE_LR, betas=(cfg.SOLVER.ADAM.B1, cfg.SOLVER.ADAM.B2))
        logging.getLogger(__name__).info("select {}/{} as trainer model".format(cfg.MODEL.ARCH, self.__class__))
        return

    def __call__(self, x, gt, epoch=None):
        real_A = x.to(self.device)
        real_B = gt.to(self.device)

        self.optimizer_G.zero_grad()

        fake_B, weights_norm = self.generator(real_A)

        loss_pixel = self.criterion_pixelwise(fake_B, real_B)

        tv0, mn0 = self.tv3(self.lut0)
        tv1, mn1 = self.lut1.tv(self.tv3)

        tv_cons = sum(tv1) + tv0
        mn_cons = sum(mn1) + mn0

        loss = loss_pixel + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        return {'psnr_avg':psnr_avg, 'mse_avg':loss_pixel.item(), 'tv_cons':tv_cons.item(), 'mn_cons':mn_cons.item(), 'weights_norm':weights_norm.item()}

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

    @staticmethod
    def __get_model_state_dict(model):
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            return model.module.state_dict()
        return model.state_dict()

    @staticmethod
    def __load_model_state_dict(model, state_dict):
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    def get_state_dict(self):
        models = dict()
        models['lut'] = {0: self.__get_model_state_dict(self.lut0)}
        models['lut'].update(self.lut1.state_dict(offset=1))
        models['cls'] = self.__get_model_state_dict(self.classifier)
        return models

    def load_state_dict(self, state_dict: dict):
        self.__load_model_state_dict(self.lut0, state_dict['lut'][0])
        self.lut1.load_state_dict(state_dict['lut'], offset=1)
        self.__load_model_state_dict(self.classifier, state_dict['cls'])
        return

    def get_addition_state_dict(self):
        addition = dict()
        addition['opt_g'] = self.__get_model_state_dict(self.optimizer_G)
        return addition

    def load_addition_state_dict(self, state_dict: dict):
        self.__load_model_state_dict(self.optimizer_G, state_dict['opt_g'])
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
