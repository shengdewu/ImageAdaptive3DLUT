import math
from models.build import MODEL_ARCH_REGISTRY
from models.AdaptivePerceptualPairedModel import AdaptivePerceptualPairedModel
from models.dce.dce_loss import *
import torch


@MODEL_ARCH_REGISTRY.register()
class AdaptiveDCEPairedModel(AdaptivePerceptualPairedModel):
    def __init__(self, cfg):
        super(AdaptiveDCEPairedModel, self).__init__(cfg)

        self.spa_loss = SpatialConsistencyLoss().to(self.device)
        self.col_loss = ColorConstancyLoss().to(self.device)
        self.exp_loss = ExposureControlLoss().to(self.device)

        self.lambda_spa = cfg.SOLVER.LAMBDA_SPA
        self.lambda_exp = cfg.SOLVER.LAMBDA_EXP
        self.lambda_col = cfg.SOLVER.LAMBDA_COL

        return

    def __call__(self, x, gt, epoch=None):
        real_A = x.to(self.device, non_blocking=True)
        real_B = gt.to(self.device, non_blocking=True)

        self.optimizer_G.zero_grad()

        fake_B, weights_norm = self.generator(real_A)

        loss_pixel = self.criterion_pixelwise(fake_B, real_B)

        tv0, mn0 = self.tv3(self.lut0)
        tv1, mn1 = self.lut1.tv(self.tv3)

        tv_cons = sum(tv1) + tv0
        mn_cons = sum(mn1) + mn0

        loss_perceptual = self.criterion_perceptual(fake_B, real_B)

        spa_loss = torch.mean(self.spa_loss(real_B, fake_B))
        col_loss = torch.mean(self.col_loss(fake_B))
        exp_loss = self.exp_loss(fake_B)

        dce_loss = self.lambda_spa * spa_loss + self.lambda_col * col_loss + self.lambda_exp * exp_loss

        lut_loss = self.lambda_pixel * loss_pixel + self.lambda_perceptual * loss_perceptual + self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + self.lambda_monotonicity * mn_cons

        loss = dce_loss + lut_loss

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        self.scheduler.step(epoch=epoch)

        learing_rate = '*'.join([str(lr) for lr in self.scheduler.get_last_lr()])

        return {'psnr_avg':psnr_avg,
                'total_loss': loss.item(),
                'mse_avg':loss_pixel.item(),
                'perceptual':loss_perceptual.item(),
                'tv_cons':tv_cons.item(),
                'mn_cons':mn_cons.item(),
                'weights_norm':weights_norm.item(),
                'spa_loss': spa_loss.item(),
                'col_loss': col_loss.item(),
                'exp_loss': exp_loss.item(),
                'learing_rate': learing_rate}
