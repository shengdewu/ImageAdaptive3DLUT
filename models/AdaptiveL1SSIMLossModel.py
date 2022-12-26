import math
from models.build import MODEL_ARCH_REGISTRY
from models.AdaptiveBaseModel import AdaptiveBaseModel
import torch
import engine.loss.ssim_loss as engine_ssim
from engine.loss.vgg_loss import PerceptualLoss


@MODEL_ARCH_REGISTRY.register()
class AdaptiveL1SSIMLossModel(AdaptiveBaseModel):
    def __init__(self, cfg):
        super(AdaptiveL1SSIMLossModel, self).__init__(cfg)
        self.ssim_loss = engine_ssim.SSIM()
        self.criterion_pixelwise = torch.nn.L1Loss().to(self.device)
        # self.criterion_perceptual = PerceptualLoss(cfg.MODEL.VGG.VGG_LAYER, device=self.device, path=cfg.MODEL.VGG.VGG_PATH)
        # self.cos_loss = torch.nn.CosineSimilarity(dim=1)

        self.lambda_ssim = cfg.SOLVER.LAMBDA_SSIM
        self.lambda_perceptual = cfg.SOLVER.LAMBDA_PERCEPTUAL
        self.lambda_pixel = cfg.SOLVER.LAMBDA_PIXEL
        self.lambda_cos = cfg.SOLVER.LAMBDA_COS
        return

    def __call__(self, x, gt, epoch=None):
        real_A = x.to(self.device, non_blocking=True)
        real_B = gt.to(self.device, non_blocking=True)

        self.optimizer_G.zero_grad()

        fake_B, weights_norm = self.generator(real_A)

        loss_pixel = self.criterion_pixelwise(fake_B, real_B)
        ssim_loss = 1.0 - self.ssim_loss(fake_B, real_B)

        tv0, mn0 = self.tv3(self.lut0)
        tv1, mn1 = self.lut1.tv(self.tv3)

        tv_cons = sum(tv1) + tv0
        mn_cons = sum(mn1) + mn0

        # loss_perceptual = self.criterion_perceptual(fake_B, real_B)
        # cos_loss = 1 - torch.mean(self.cos_loss(fake_B, real_B))

        # loss = self.lambda_pixel * loss_pixel + self.lambda_perceptual * loss_perceptual + \
        #        self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + \
        #        self.lambda_monotonicity * mn_cons + self.lambda_ssim * ssim_loss + self.lambda_cos * cos_loss

        loss = self.lambda_pixel * loss_pixel + self.lambda_ssim * ssim_loss + \
               self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + \
               self.lambda_monotonicity * mn_cons

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        self.scheduler.step(epoch=epoch)

        learing_rate = '*'.join([str(lr) for lr in self.scheduler.get_last_lr()])

        return {'psnr_avg': psnr_avg,
                'total_loss': loss.item(),
                'loss_pixel': loss_pixel.item(),
                'ssim_loss': ssim_loss.item(),
                # 'perceptual': loss_perceptual.item(),
                # 'cos_loss': cos_loss.item(),
                'mn_cons': mn_cons.item(),
                'tv_cons': tv_cons.item(),
                'weights_norm': weights_norm.item(),
                'learing_rate': learing_rate}
