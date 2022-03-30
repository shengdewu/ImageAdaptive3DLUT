import math
from models.build import MODEL_ARCH_REGISTRY
from models.AdaptivePerceptualPairedModel import AdaptivePerceptualPairedModel
import torch
import torchvision.transforms as tt


@MODEL_ARCH_REGISTRY.register()
class AdaptiveGaussianLossModel(AdaptivePerceptualPairedModel):
    def __init__(self, cfg):
        super(AdaptiveGaussianLossModel, self).__init__(cfg)
        self.lambda_col = cfg.SOLVER.LAMBDA_COL
        self.gauss = tt.GaussianBlur(kernel_size=21, sigma=3.0)
        self.color_loss = torch.nn.MSELoss()
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

        color_loss = self.color_loss(self.gauss(fake_B), self.gauss(real_B))

        loss = self.lambda_pixel * loss_pixel + self.lambda_perceptual * loss_perceptual + \
               self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + \
               self.lambda_monotonicity * mn_cons + self.lambda_col * color_loss

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        self.scheduler.step(epoch=epoch)

        learing_rate = '*'.join([str(lr) for lr in self.scheduler.get_last_lr()])

        return {'psnr_avg': psnr_avg,
                'total_loss': loss.item(),
                'loss_pixel': loss_pixel.item(),
                'loss_perceptual': loss_perceptual.item(),
                'color_loss': color_loss.item(),
                'mn_cons': mn_cons.item(),
                'tv_cons': tv_cons.item(),
                'weights_norm': weights_norm.item(),
                'learing_rate': learing_rate}
