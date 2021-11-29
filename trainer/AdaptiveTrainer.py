import logging
from dataloader.dataloader import DataLoader
from engine.checkpoint.CheckpointerManager import CheckpointerManager
from models import build_model
import engine.comm as comm
from engine.log.logger import setup_logger
import torch
import math
import torchvision


class AdaptiveTrainer:
    criterion_pixelwise = torch.nn.MSELoss()

    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        self.model = build_model(cfg)
        self.model.enable_train()

        train_dataset, test_dataset = DataLoader.create_dataset(cfg)
        logging.getLogger(__name__).info('create dataset {}  then load {} train data, load {} test data'.format(cfg.DATALOADER.DATASET, len(train_dataset), len(test_dataset)))

        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            self.dataloader = DataLoader.create_distribute_sampler_dataloder(train_dataset,
                                                                             cfg.SOLVER.IMS_PER_BATCH,
                                                                             cfg.MODEL.TRAINER.GLOBAL_RANK,
                                                                             cfg.MODEL.TRAINER.WORLD_SIZE,
                                                                             cfg.DATALOADER.NUM_WORKERS)
        else:
            self.dataloader = DataLoader.create_sampler_dataloader(train_dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS)

        self.test_dataloader = DataLoader.create_dataloader(test_dataset)

        self.model.enable_distribute(cfg)

        self.checkpoint = CheckpointerManager(max_iter=cfg.SOLVER.MAX_ITER,
                                              save_dir=cfg.OUTPUT_DIR,
                                              check_period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                              max_keep=cfg.SOLVER.MAX_KEEP,
                                              file_prefix=cfg.MODEL.ARCH,
                                              save_to_disk=comm.is_main_process())

        self.device = cfg.MODEL.DEVICE
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.model_path = cfg.MODEL.WEIGHTS
        self.iter_train_loader = iter(self.dataloader)
        self.output = cfg.OUTPUT_DIR
        logging.getLogger(__name__).info('ready for training : there are {} data in one epoch and actually trained for {} epoch'.format(len(train_dataset) / cfg.SOLVER.IMS_PER_BATCH, self.max_iter / (len(train_dataset) / cfg.SOLVER.IMS_PER_BATCH)))
        return

    def loop(self):
        psnr = self.calculate_psnr(self.model, self.test_dataloader, self.device)
        logging.getLogger(__name__).info('before train psnr = {}'.format(psnr))

        total_cnt = 0
        loss_avg = dict()
        for epoch in range(self.start_iter, self.max_iter):
            data = next(self.iter_train_loader)
            gt = data['A_exptC'] if 'B_exptC' not in data.keys() else data['B_exptC']
            loss = self.model(data['A_input'], gt, epoch)

            total_cnt += 1.0
            for k, v in loss.items():
                loss_avg[k] = loss_avg.get(k, 0) + v

            addtion = self.model.get_addition_state_dict()
            self.checkpoint.save(epoch, self.model.get_state_dict(), **addtion)
            self.run_after(epoch, loss_avg, total_cnt)

        addtion = self.model.get_addition_state_dict()
        self.checkpoint.save(self.max_iter, self.model.get_state_dict(), **addtion)

        psnr = self.calculate_psnr(self.model, self.test_dataloader, self.device)
        logging.getLogger(__name__).info('after train psnr = {}'.format(psnr))
        self.visualize_result(self.model, self.test_dataloader, self.device, self.output)
        return

    def run_after(self, epoch, loss_avg, total_cnt):
        if epoch % self.checkpoint.check_period == 0:
            loss_str = ''
            for k, v in loss_avg.items():
                if len(loss_str) > 0:
                    loss_str += ', '
                loss_str += '{}:{}'.format(k, v / total_cnt)
            logging.getLogger(__name__).info('trainer run step {} : {}'.format(epoch, loss_str))
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
    def calculate_psnr(trainer, dataloader, device):
        trainer.disable_train()
        avg_psnr = 0
        for i, batch in enumerate(dataloader):
            real_A = batch["A_input"].to(device)
            real_B = batch["A_exptC"].to(device)
            fake_B, weights_norm = trainer.generator(real_A)
            fake_B = torch.round(fake_B * 255)
            real_B = torch.round(real_B * 255)
            mse = AdaptiveTrainer.criterion_pixelwise(fake_B, real_B)
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
            mse = AdaptiveTrainer.criterion_pixelwise(fake_B, real_B)
            mse = torch.clip(mse, 0.00000001, 4294967296.0)
            psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
            torchvision.utils.save_image(img_sample, '{}/{}-{}.jpg'.format(save_path, i, str(psnr)[:5]), nrow=1, normalize=False)
        return
