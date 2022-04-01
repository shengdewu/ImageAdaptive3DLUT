import torch
import math
from models.build import MODEL_ARCH_REGISTRY
from models.AdaptiveBaseModel import AdaptiveBaseModel


@MODEL_ARCH_REGISTRY.register()
class AdaptivePairedModel(AdaptiveBaseModel):
    def __init__(self, cfg):
        super(AdaptivePairedModel, self).__init__(cfg)

        self.lambda_pixel = cfg.SOLVER.LAMBDA_PIXEL
        # Loss functions
        self.criterion_pixelwise = torch.nn.MSELoss().to(self.device)
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

        loss = self.lambda_pixel * loss_pixel + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        return {'psnr_avg':psnr_avg, 'mse_avg':loss_pixel.item(), 'tv_cons':tv_cons.item(), 'mn_cons':mn_cons.item(), 'weights_norm':weights_norm.item()}
