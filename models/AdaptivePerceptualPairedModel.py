import math
from models.build import MODEL_ARCH_REGISTRY
from models.AdaptivePairedModel import AdaptivePairedModel
from engine.log.logger import setup_logger
import engine.comm as comm
from models.vgg_loss import PerceptualLoss


@MODEL_ARCH_REGISTRY.register()
class AdaptivePerceptualPairedModel(AdaptivePairedModel):
    def __init__(self, cfg):
        super(AdaptivePerceptualPairedModel, self).__init__(cfg)
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        self.lambda_pixel = cfg.SOLVER.LAMBDA_PIXEL
        self.lambda_perceptual = cfg.SOLVER.LAMBDA_PERCEPTUAL
        self.lambda_class_smooth = cfg.SOLVER.LAMBDA_CLASS_SMOOTH

        self.criterion_perceptual = PerceptualLoss(cfg.MODEL.VGG.VGG_LAYER, path=cfg.MODEL.VGG.VGG_PATH)

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

        loss = self.lambda_pixel * loss_pixel + self.lambda_perceptual * loss_perceptual + self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + self.lambda_monotonicity * mn_cons

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        loss.backward()

        self.optimizer_G.step()

        self.scheduler.step(epoch=epoch)

        learing_rate = '*'.join([str(lr) for lr in self.scheduler.get_last_lr()])

        return {'psnr_avg':psnr_avg,
                'mse_avg':loss_pixel.item(),
                'perceptual':loss_perceptual.item(),
                'tv_cons':tv_cons.item(),
                'mn_cons':mn_cons.item(),
                'weights_norm':weights_norm.item(),
                'learing_rate': learing_rate}
