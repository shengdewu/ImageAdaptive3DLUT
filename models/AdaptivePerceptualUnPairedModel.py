import torch
import math
from models.build import MODEL_ARCH_REGISTRY
from models.vgg_loss import PerceptualLoss
from models.functional import compute_gradient_penalty
from models.AdaptiveUnpairedModel import AdaptiveUnPairedModel
from engine.log.logger import setup_logger
import engine.comm as comm


@MODEL_ARCH_REGISTRY.register()
class AdaptivePerceptualUnPairedModel(AdaptiveUnPairedModel):
    def __init__(self, cfg):
        super(AdaptivePerceptualUnPairedModel, self).__init__(cfg)
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        self.lambda_perceptual = cfg.SOLVER.LAMBDA_PERCEPTUAL

        self.criterion_perceptual = PerceptualLoss(cfg.MODEL.VGG.VGG_LAYER, path=cfg.MODEL.VGG.VGG_PATH)
        return

    def __call__(self, x, gt, epoch=None):
        real_A = x.to(self.device, non_blocking=True)
        real_B = gt.to(self.device, non_blocking=True)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()

        fake_B, weights_norm = self.generator(real_A)

        pred_real = self.discriminator(real_B)
        pred_fake = self.discriminator(fake_B)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(self.discriminator, real_B, fake_B, grad_outputs_shape=pred_real.shape, device=self.device)

        # Total loss
        loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) + self.lambda_gp * gradient_penalty

        loss_D.backward()
        self.optimizer_D.step()

        loss_D_avg = (-torch.mean(pred_real) + torch.mean(pred_fake)) / 2
        psnr_avg = 0
        # ------------------
        #  Train Generators
        # ------------------
        loss_G_avg = torch.zeros(1)
        loss_pixel = torch.zeros(1)
        loss_perceptual = torch.zeros(1)
        tv_cons = 0.0
        mn_cons = 0.0
        if epoch % self.n_critic == 0:
            self.optimizer_G.zero_grad()

            fake_B, weights_norm = self.generator(real_A)
            pred_fake = self.discriminator(fake_B)
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_A)

            loss_perceptual = self.criterion_perceptual(fake_B, real_B)

            tv0, mn0 = self.tv3(self.lut0)
            tv1, mn1 = self.lut1.tv(self.tv3)

            tv_cons = sum(tv1) + tv0
            mn_cons = sum(mn1) + mn0

            loss_G = -torch.mean(pred_fake) + self.lambda_pixel * loss_pixel + self.lambda_perceptual * loss_perceptual + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

            loss_G.backward()

            self.optimizer_G.step()

            loss_G_avg = -torch.mean(pred_fake)
            psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        return {'loss_D_avg': loss_D_avg.item(), 'loss_G_avg': loss_G_avg.item(), 'loss_pixel_avg': loss_pixel.item(), 'loss_perceptual': loss_perceptual.item(), 'psnr_avg': psnr_avg,
                'tv_cons': tv_cons.item(), 'mn_cons': mn_cons.item(), 'weights_norm': weights_norm.item()}
