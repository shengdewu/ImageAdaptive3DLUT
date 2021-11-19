import logging
import argparse
import os
import sys
import torch
import math
from trainer import build_trainer
from dataloader.dataloader import DataLoader
import torchvision.utils


class Adaptive3Dlut:

    criterion_pixelwise = torch.nn.MSELoss()

    def __init__(self):
        return

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--trainer", type=str, default='TrainerUnPaired', help="trainer type")
        parser.add_argument("--classifier", type=str, default='ClassifierResnet', help="classifier type [Classifier, ClassifierResnet]")
        parser.add_argument("--pretrain", action="store_true", help="whether or not load model")
        parser.add_argument("--lut_dim", type=int, default=33, help="dim of the lut")
        parser.add_argument("--lut_nums", type=int, default=2, help="number of the lut")
        parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
        parser.add_argument("--n_epochs", type=int, default=400, help="total number of epochs of training")
        parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--data_path", type=str, default='', help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--gamma", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--step_lr_epoch", type=str, default='10, 20', help="learning rate decay step")
        parser.add_argument("--use_cpu", action="store_true", help="use cpu or gpu")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--lambda_pixel", type=float, default=1000, help="content preservation weight: 1000 for sRGB input, 10 for XYZ input")
        parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty weight in wgan-gp")
        parser.add_argument("--lambda_smooth", type=float, default=1e-4, help="smooth regularization")
        parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization: 10 for sRGB input, 100 for XYZ input (slightly better)")
        parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
        parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
        parser.add_argument("--output_dir", type=str, default="./", help="path to save model")
        parser.add_argument("--classifier_model_path", type=str, default='', help="model path of classifier")
        parser.add_argument("--model_name", type=str, default="", help="path to save model")
        parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
        parser.add_argument("--max_checkpoints", type=int, default=5, help="number of model checkpoints")
        parser.add_argument("--visualize_batch", type=int, default=100, help="start epoch for visualize")
        opt = parser.parse_args()

        opt.step_lr_epoch = [int(v.strip()) for v in opt.step_lr_epoch.split(',')]
        opt.unpaired = False
        if opt.trainer == 'TrainerUnPaired':
            opt.unpaired = True
        opt.device = 'cpu' if opt.use_cpu else 'cuda'
        return opt

    @staticmethod
    def init_log(log_name, log_path):
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        logging.basicConfig(
            filename=log_path + '/' + log_name + '.log',
            format='<%(levelname)s %(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s> %(message)s',
            level=logging.INFO)

        log = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        log.addHandler(stdout_handler)
        return

    @staticmethod
    def create_trainer(cfg):
        trainer = build_trainer(cfg)
        if cfg.pretrain:
            trainer.load_model(cfg.output_dir, cfg.model_name)
        return trainer

    @staticmethod
    @torch.no_grad()
    def calculate_psnr(trainer, dataloader, device):
        trainer.disable_train()
        avg_psnr = 0
        for i, batch in enumerate(dataloader):
            real_A = batch["A_input"].to(device)
            real_B = batch["A_exptC"].to(device)
            fake_B, weights_norm = trainer.generator(real_A)
            fake_B = torch.round(fake_B * 255)
            real_B = torch.round(real_B * 255)
            mse = Adaptive3Dlut.criterion_pixelwise(fake_B, real_B)
            mse = torch.clip(mse, 0.00000001, 4294967296.0)
            psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
            avg_psnr += psnr

        return avg_psnr / len(dataloader)

    @staticmethod
    @torch.no_grad()
    def visualize_result(trainer, dataloader, device, save_path):
        trainer.disable_train()
        for i, batch in enumerate(dataloader):
            real_A = batch["A_input"].to(device)
            real_B = batch["A_exptC"].to(device)
            fake_B, weights_norm = trainer.generator(real_A)
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
            fake_B = torch.round(fake_B * 255)
            real_B = torch.round(real_B * 255)
            mse = Adaptive3Dlut.criterion_pixelwise(fake_B, real_B)
            mse = torch.clip(mse, 0.00000001, 4294967296.0)
            psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
            torchvision.utils.save_image(img_sample, '{}/{}-{}.jpg'.format(save_path, i, str(psnr)[:5]), nrow=1, normalize=False)
        return

    @staticmethod
    def check_point(trainer, output_dir, max_checkpoints, epoch):
        trainer.save_model(output_dir, epoch=epoch)
        luts_name, cls_name = trainer.get_name_prefix()

        luts_name_latest = '{}_{}.pth'.format(luts_name, 'latest')
        cls_name_latest = '{}_{}.pth'.format(cls_name, 'latest')

        luts_model_list = [name for name in os.listdir(output_dir) if name.startswith(luts_name) and name != luts_name_latest]
        luts_model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(output_dir, fn)))

        cls_model_list = [name for name in os.listdir(output_dir) if name.startswith(cls_name) and name != cls_name_latest]
        cls_model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(output_dir, fn)))

        assert len(cls_model_list) == len(luts_model_list)
        if len(luts_model_list) > max_checkpoints:
            for luts_model in luts_model_list[:len(luts_model_list)-max_checkpoints]:
                if os.path.exists(os.path.join(output_dir, luts_model)):
                    os.remove(os.path.join(output_dir, luts_model))
            for cls_model in cls_model_list[:len(cls_model_list)-max_checkpoints]:
                if os.path.exists(os.path.join(output_dir, cls_model)):
                    os.remove(os.path.join(output_dir, cls_model))
        return

    @staticmethod
    def loop():
        cfg = Adaptive3Dlut.parse()
        Adaptive3Dlut.init_log(__name__, cfg.output_dir)
        logging.info('load cfg : {}'.format(cfg))

        trainer = Adaptive3Dlut.create_trainer(cfg)
        dataloader, psnr_dataloader, test_dataset = DataLoader.create_dataloader(cfg)
        total_train = len(dataloader)

        psnr = Adaptive3Dlut.calculate_psnr(trainer, psnr_dataloader, cfg.device)
        logging.info('befor train avg psnr in all test data = {}'.format(psnr))

        loss_avg = dict()
        total_cnt = 0.0

        for epoch in range(cfg.epoch, cfg.n_epochs):
            trainer.enable_train()

            for idx, x in enumerate(dataloader):
                gt = x['A_exptC'] if 'B_exptC' not in x.keys() else x['B_exptC']
                loss = trainer.step(x['A_input'], gt, idx)

                total_cnt += 1.0
                for k, v in loss.items():
                    loss_avg[k] = loss_avg.get(k, 0) + v

                train_step = epoch * total_train + idx
                if (train_step + 1) % cfg.checkpoint_interval == 0:
                    Adaptive3Dlut.check_point(trainer, cfg.output_dir, cfg.max_checkpoints, epoch=epoch)
                    loss_str = ''
                    for k, v in loss_avg.items():
                        if len(loss_str) > 0:
                            loss_str += ', '
                        loss_str += '{}:{}'.format(k, v/total_cnt)

                    logging.info('epoch {}-iter {} : loss {} '.format(epoch, idx, loss_str))

            Adaptive3Dlut.check_point(trainer, cfg.output_dir, cfg.max_checkpoints, epoch='latest')

        Adaptive3Dlut.visualize_result(trainer, psnr_dataloader, cfg.device, cfg.output_dir)
        psnr = Adaptive3Dlut.calculate_psnr(trainer, psnr_dataloader, cfg.device)
        logging.info('after train avg psnr in all test data = {}'.format(psnr))

        return


