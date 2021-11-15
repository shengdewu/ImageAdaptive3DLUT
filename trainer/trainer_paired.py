import torch
import itertools
import os
import math
from trainer import TRAINER_ARCH_REGISTRY
from models.lut.generator_3dlut import Generator_3DLUT_identity, Generator_3DLUT_n_zero
from models.classifier.classifier import Classifier
from models.lut.total_variation import TV_3D
from models.functional import weights_init_normal_classifier
from .trainer_base import TrainerBase


@TRAINER_ARCH_REGISTRY.register()
class TrainerPaired(TrainerBase):
    def __init__(self, cfg):
        super(TrainerPaired, self).__init__(cfg)
        self.luts_name_prefix = 'luts_paired'
        self.cls_name_prefix = 'classifier_paired'

        self.lut0 = Generator_3DLUT_identity(cfg.lut_dim, cfg.device)
        self.lut1 = Generator_3DLUT_n_zero(cfg.lut_dim, cfg.lut_nums, cfg.device)
        self.classifier = Classifier(device=cfg.device, class_num=cfg.lut_nums+1)
        self.tv3 = TV_3D(cfg.lut_dim, cfg.device)

        self.classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(self.classifier.model[12].bias.data, 1.0) # last layer

        # Loss functions
        self.criterion_pixelwise = torch.nn.MSELoss()

        parameters = self.lut1.parameters()
        parameters.insert(0, self.lut0.parameters())
        parameters.insert(0, self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(itertools.chain(*parameters), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        return

    def step(self, x, gt, epoch=None):
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

        assert pred.shape[0] - 1 == len(self.lut1)

        if len(pred.shape) == 1:
            pred = pred.unsqueeze(0)

        gen_a = self.lut0(img)
        gen_a1 = self.lut1(img)

        combine_a = img.new(img.size())
        for b in range(img.size(0)):
            combine_a[b, :, :, :] = pred[b, 0] * gen_a[b, :, :, :]
            for i, val in gen_a1.items():
                combine_a[b, :, :, :] = pred[b, i+1] * val[b, :, :, :]

        weights_norm = torch.mean(pred ** 2)
        return combine_a, weights_norm

