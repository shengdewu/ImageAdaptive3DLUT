import logging
from dataloader.dataloader import DataLoader
from engine.checkpoint.checkpoint_manager import CheckPointerManager
from models.build import build_model
import engine.comm as comm
from engine.log.logger import setup_logger
import torch
import math
import torchvision
import cv2
import numpy as np
from engine.data.data_loader import create_distribute_iterable_data_loader, create_iterable_data_loader, create_data_loader


class AdaptiveTrainer:
    criterion_pixelwise = torch.nn.MSELoss()

    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        self.model = build_model(cfg)
        self.model.enable_train()

        train_dataset, test_dataset = DataLoader.create_dataset(cfg)
        logging.getLogger(__name__).info('create dataset {}  then load {} train data, load {} test data'.format(cfg.DATALOADER.DATASET, len(train_dataset), len(test_dataset)))

        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            self.dataloader = create_distribute_iterable_data_loader(dataset=train_dataset,
                                                                     batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                                     rank=cfg.MODEL.TRAINER.GLOBAL_RANK,
                                                                     world_size=cfg.MODEL.TRAINER.WORLD_SIZE,
                                                                     num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                     collate_fn=DataLoader.collate_fn)
        else:
            self.dataloader = create_iterable_data_loader(dataset=train_dataset,
                                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                          num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                          collate_fn=DataLoader.collate_fn)

        self.test_dataloader = create_data_loader(dataset=test_dataset, batch_size=cfg.SOLVER.TEST_PER_BATCH, num_workers=1, collate_fn=DataLoader.collate_fn)

        self.model.enable_distribute(cfg)

        self.checkpoint = CheckPointerManager(max_iter=cfg.SOLVER.MAX_ITER,
                                              save_dir=cfg.OUTPUT_DIR,
                                              check_period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                              max_keep=cfg.SOLVER.MAX_KEEP,
                                              file_prefix=cfg.MODEL.ARCH,
                                              save_to_disk=comm.is_main_process())

        self.device = cfg.MODEL.DEVICE
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.model_path = cfg.MODEL.WEIGHTS
        self.output = cfg.OUTPUT_DIR
        self.total_data_per_epoch = len(train_dataset) / cfg.SOLVER.IMS_PER_BATCH
        self.iter_train_loader = iter(self.dataloader)
        logging.getLogger(__name__).info('ready for training : there are {} data in one epoch and actually trained for {} epoch'.format(self.total_data_per_epoch, self.max_iter / self.total_data_per_epoch))
        return

    def loop(self):
        psnr = self.calculate_psnr(self.model, self.test_dataloader, self.device)
        logging.getLogger(__name__).info('before train psnr = {}'.format(psnr))

        self.model.enable_train()

        for epoch in range(self.start_iter, self.max_iter):
            data = next(self.iter_train_loader)

            gt = data['A_exptC'] if 'B_exptC' not in data.keys() else data['B_exptC']
            loss_dict = self.model(data['A_input'], gt, epoch)
            self.checkpoint.save(self.model, epoch)
            self.run_after(epoch, loss_dict)

        self.checkpoint.save(self.model, self.max_iter)

        psnr = self.calculate_psnr(self.model, self.test_dataloader, self.device)
        logging.getLogger(__name__).info('after train psnr = {}'.format(psnr))
        self.visualize_result(self.model, self.test_dataloader, self.device, self.output)
        return

    def run_after(self, epoch, loss_dict):
        if int(epoch+0.5) % self.checkpoint.check_period == 0:
            logging.getLogger(__name__).info('trainer run step {} : {}'.format(epoch, loss_dict))
        return

    def resume_or_load(self, resume=False):
        model_state_dict, addition_state_dict, start_iter = self.checkpoint.resume_or_load(self.model_path, resume)
        self.start_iter = start_iter
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if addition_state_dict is not None:
            self.model.load_addition_state_dict(addition_state_dict)
        logging.getLogger(__name__).info('load model from {}: resume:{} start iter:{}'.format(self.model_path, resume, self.start_iter))
        return

    @staticmethod
    @torch.no_grad()
    def calculate_psnr(trainer, dataloader, device, unnormalizing_value=255):
        trainer.disable_train()
        avg_psnr = 0
        for i, batch in enumerate(dataloader):
            real_A = batch["A_input"].to(device)
            real_B = batch["A_exptC"].to(device)
            fake_B, weights_norm = trainer.generator(real_A)
            fake_B = torch.round(fake_B * unnormalizing_value)
            real_B = torch.round(real_B * unnormalizing_value)
            mse = AdaptiveTrainer.criterion_pixelwise(fake_B, real_B)
            mse = torch.clip(mse, 0.00000001, 4294967296.0)
            psnr = 10.0 * math.log10(float(unnormalizing_value) * unnormalizing_value / mse.item())
            avg_psnr += psnr

        return avg_psnr / len(dataloader)

    @staticmethod
    @torch.no_grad()
    def visualize_result(trainer, dataloader, device, save_path, unnormalizing_value=255):
        trainer.disable_train()
        format = 'jpg' if unnormalizing_value == 255 else 'tif'
        for i, batch in enumerate(dataloader):
            real_A = batch["A_input"].to(device)
            real_B = batch["A_exptC"].to(device)
            fake_B, weights_norm = trainer.generator(real_A)
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
            fake_B = torch.round(fake_B * unnormalizing_value)
            real_B = torch.round(real_B * unnormalizing_value)
            mse = AdaptiveTrainer.criterion_pixelwise(fake_B, real_B)
            mse = torch.clip(mse, 0.00000001, 4294967296.0)
            psnr = 10.0 * math.log10(float(unnormalizing_value) * unnormalizing_value / mse.item())
            AdaptiveTrainer.save_image(img_sample, '{}/{}-{}.{}'.format(save_path, i, str(psnr)[:5], format), unnormalizing_value=unnormalizing_value, nrow=1, normalize=False)
        return

    @staticmethod
    @torch.no_grad()
    def save_image(tensor, fp, unnormalizing_value=255, **kwargs):
        fmt = np.uint8 if unnormalizing_value == 255 else np.uint16
        grid = torchvision.utils.make_grid(tensor, **kwargs)
        # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
        ndarr = grid.mul(unnormalizing_value).add_(0.5).clamp_(0, unnormalizing_value).permute(1, 2, 0).to('cpu').numpy().astype(fmt)
        cv2.imwrite(fp, ndarr[:, :, ::-1])
        return
