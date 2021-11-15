import torch
import itertools
import math
from trainer import TRAINER_ARCH_REGISTRY
from models.lut.generator_3dlut import Generator_3DLUT_identity, Generator_3DLUT_n_zero
from models.classifier.classifier import Classifier_unpaired
from models.discriminator.discriminator import Discriminator
from models.lut.total_variation import TV_3D
from models.functional import weights_init_normal_classifier, compute_gradient_penalty
from .trainer_base import TrainerBase


@TRAINER_ARCH_REGISTRY.register()
class TrainerUnPaired(TrainerBase):
    def __init__(self, cfg):
        super(TrainerUnPaired, self).__init__(cfg)
        self.lambda_gp = cfg.lambda_gp
        self.lambda_pixel = cfg.lambda_pixel
        self.n_critic = cfg.n_critic
        self.model_prefix = 'unpaired'

        self.lut0 = Generator_3DLUT_identity(cfg.lut_dim, cfg.device)
        self.lut1 = Generator_3DLUT_n_zero(cfg.lut_dim, cfg.lut_nums, cfg.device)
        self.classifier = Classifier_unpaired(device=cfg.device, class_num=cfg.lut_nums + 1)
        self.discriminator = Discriminator(device=cfg.device)
        self.tv3 = TV_3D(cfg.lut_dim, cfg.device)

        self.classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(self.classifier.model[12].bias.data, 1.0)  # last layer
        self.discriminator.apply(weights_init_normal_classifier)

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.MSELoss()

        parameters = self.lut1.parameters()
        parameters.insert(0, self.lut0.parameters())
        parameters.insert(0, self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(itertools.chain(*parameters), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        return

    def step(self, x, gt, epoch):
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

    def generator(self, img):
        pred = self.classifier(img).squeeze()

        assert pred.shape[0] - 1 == len(self.lut1)

        combine_a = pred[0] * self.lut0(img)
        lut1 = self.lut1(img)
        for i, val in lut1.items():
            combine_a += pred[i + 1] * val

        weights_norm = torch.mean(pred ** 2)
        return combine_a, weights_norm

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

