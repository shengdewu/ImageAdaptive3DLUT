import torch
import os
import logging
from trainer import TRAINER_ARCH_REGISTRY


@TRAINER_ARCH_REGISTRY.register()
class TrainerBase:
    def __init__(self, cfg):
        self.device = cfg.device
        self.step_lr_epoch = cfg.step_lr_epoch
        self.gamma = cfg.gamma
        self.base_lr = cfg.lr
        self.tmp_lr = self.base_lr
        self.lambda_smooth = cfg.lambda_smooth
        self.lambda_monotonicity = cfg.lambda_monotonicity

        self.lut0 = None
        self.lut1 = None
        self.classifier = None
        self.model_prefix = ''

        return

    def save_model(self, save_path, epoch):
        luts = {0: self.lut0.state_dict()}
        luts.update(self.lut1.state_dict(offset=1))
        torch.save(luts, os.path.join(save_path, 'luts_{}_{}.pth'.format(self.model_prefix, epoch)))
        torch.save(self.classifier.state_dict(), os.path.join(save_path, 'classifier_{}_{}.pth'.format(self.model_prefix, epoch)))
        return

    def load_model(self, model_path, model_name):
        if model_name == '':
            model_list = [name for name in os.listdir(model_path) if name.startswith('luts_{}'.format(self.model_prefix))]
            model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(model_path, fn)))
            lut_name = os.path.join(model_path, model_list[-1])

            model_list = [name for name in os.listdir(model_path) if name.startswith('classifier_{}'.format(self.model_prefix))]
            model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(model_path, fn)))
            class_name = os.path.join(model_path, model_list[-1])
        else:
            model_name = model_name.split(',')
            lut_name = os.path.join(model_path, model_name[0].strip())
            class_name = os.path.join(model_path, model_name[1].strip())

        logging.info('load class model {}'.format(class_name))
        logging.info('load lut model {}'.format(lut_name))

        if self.device == 'cpu':
            luts_state_dict = torch.load(lut_name, map_location='cpu')
            class_state_dict = torch.load(class_name, map_location='cpu')
        else:
            luts_state_dict = torch.load(lut_name)
            class_state_dict = torch.load(class_name)

        self.lut0.load_state_dict(luts_state_dict[0])
        self.lut1.load_state_dict(luts_state_dict, offset=1)
        self.classifier.load_state_dict(class_state_dict)
        return

    def enable_train(self):
        self.lut0.train()
        self.lut1.train()
        self.classifier.train()
        return

    def disable_train(self):
        self.lut0.eval()
        self.lut1.eval()
        self.classifier.eval()
        return

    def set_lr(self, lr):
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        return

    def update_learn_rate_step(self, epoch):
        if self.step_lr_epoch is None:
            return

        if epoch in self.step_lr_epoch:
            self.tmp_lr = self.tmp_lr * self.gamma
            self.set_lr(self.tmp_lr)
        return self.tmp_lr
