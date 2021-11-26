import torch
import math
from models import MODEL_ARCH_REGISTRY
from models.discriminator.discriminator import Discriminator
from models.functional import weights_init_normal_classifier, compute_gradient_penalty
from models.AdaptivePairedModel import AdaptivePairedModel
import logging
from engine.log.logger import setup_logger
import engine.comm as comm
from models.functional import get_model_state_dict, load_model_state_dict


@MODEL_ARCH_REGISTRY.register()
class AdaptiveUnPairedModel(AdaptivePairedModel):
    def __init__(self, cfg):
        super(AdaptiveUnPairedModel, self).__init__(cfg)
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        self.lambda_gp = cfg.SOLVER.LAMBDA_GP
        self.lambda_pixel = cfg.SOLVER.LAMBDA_PIXEL
        self.n_critic = cfg.SOLVER.N_CRITIC
        self.luts_name_prefix = 'luts_unpaired'
        self.cls_name_prefix = 'classifier_unpaired'

        self.discriminator = Discriminator(device=cfg.MODEL.DEVICE)
        self.discriminator.apply(weights_init_normal_classifier)

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()

        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(cfg.SOLVER.ADAM.B1, cfg.SOLVER.ADAM.B2))
        return

    def __call__(self, x, gt, epoch=None):
        real_A = x.to(self.device)
        real_B = gt.to(self.device)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()

        fake_B, weights_norm = self.generator(real_A)

        pred_real = self.discriminator(real_B)
        pred_fake = self.discriminator(fake_B)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(self.discriminator, real_B, fake_B, device=self.device)

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
        tv_cons = 0.0
        mn_cons = 0.0
        if epoch % self.n_critic == 0:
            self.optimizer_G.zero_grad()

            fake_B, weights_norm = self.generator(real_A)
            pred_fake = self.discriminator(fake_B)
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_A)

            tv0, mn0 = self.tv3(self.lut0)
            tv1, mn1 = self.lut1.tv(self.tv3)

            tv_cons = sum(tv1) + tv0
            mn_cons = sum(mn1) + mn0

            loss_G = -torch.mean(pred_fake) + self.lambda_pixel * loss_pixel + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

            loss_G.backward()

            self.optimizer_G.step()

            loss_G_avg = -torch.mean(pred_fake)
            psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        return {'loss_D_avg': loss_D_avg.item(), 'loss_G_avg': loss_G_avg.item(), 'loss_pixel_avg': loss_pixel.item(), 'psnr_avg': psnr_avg,
                'tv_cons': tv_cons.item(), 'mn_cons': mn_cons.item(), 'weights_norm': weights_norm.item()}

    def enable_train(self):
        self.lut0.train()
        self.lut1.train()
        self.classifier.train()
        self.discriminator.train()
        return

    def disable_train(self):
        self.lut0.eval()
        self.lut1.eval()
        self.classifier.eval()
        self.discriminator.eval()
        return

    def get_addition_state_dict(self):
        addition = dict()
        addition['dis'] = get_model_state_dict(self.discriminator)
        addition['opt_g'] = get_model_state_dict(self.optimizer_G)
        addition['opt_d'] = get_model_state_dict(self.optimizer_D)
        return addition

    def load_addition_state_dict(self, state_dict: dict):
        load_model_state_dict(self.optimizer_G, state_dict['opt_g'])
        load_model_state_dict(self.optimizer_D, state_dict['opt_d'])
        load_model_state_dict(self.discriminator, state_dict['dis'])
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
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(__name__).info('launch model by parallel')
            self.lut0 = torch.nn.parallel.DataParallel(self.lut0)
            lut1 = dict()
            for k, v in self.lut1.foreach():
                lut1[k] = torch.nn.parallel.DataParallel(v)
            self.lut1.upate(lut1)

            self.classifier = torch.nn.parallel.DataParallel(self.classifier)
            self.discriminator = torch.nn.parallel.DataParallel(self.discriminator)
        else:
            logging.getLogger(__name__).info('launch model by singal machine')
        return
