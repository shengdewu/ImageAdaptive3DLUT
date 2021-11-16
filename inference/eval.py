import torch
import os
from models.lut.generator_3dlut import Generator_3DLUT_identity, Generator_3DLUT_n_zero
from models.classifier.classifier import Classifier, Classifier_unpaired
from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel
import dataloader.torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF
import PIL.Image as Image
import cv2
import numpy as np


class EvalPaired:
    def __init__(self, lut_dim, lut_nums, device='cpu', input_color_space='sRGB', unpaired=True):
        self.model_prefix = 'paired'
        self.device = device
        self.input_color_space = input_color_space
        self.lut0 = Generator_3DLUT_identity(lut_dim, device)
        self.lut1 = Generator_3DLUT_n_zero(lut_dim, lut_nums, device)
        if unpaired:
            self.classifier = Classifier_unpaired(device=device, class_num=lut_nums+1)
        else:
            self.classifier = Classifier(device=device, class_num=lut_nums+1)
        self.trilinear = TrilinearInterpolationModel()
        return

    def read(self, img_path):
        # read image and transform to tensor
        if self.input_color_space == 'sRGB':
            img = Image.open(img_path)
        elif self.input_color_space == 'XYZ':
            img = cv2.imread(img_path, -1)
        else:
            raise NotImplemented(self.input_color_space)

        return img.unsqueeze(0)

    def to_tensor(self, x):
        # read image and transform to tensor
        if self.input_color_space == 'sRGB':
            img = TF.to_tensor(x).to(self.device)
        elif self.input_color_space == 'XYZ':
            img = np.array(x)
            img = TF_x.to_tensor(img).to(self.device)
        else:
            raise NotImplemented(self.input_color_space)

        return img.unsqueeze(0)

    def inference(self, img_or_path):
        if isinstance(img_or_path, str):
            img_or_path = self.read(img_or_path)

        x = self.to_tensor(img_or_path)

        real_A = x.to(self.device)

        lut = self.generator_lut(real_A)

        _, out = self.trilinear(lut, real_A)

        ndarr = out.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return ndarr

    def generator_lut(self, img):
        pred = self.classifier(img).squeeze()

        assert pred.shape[0] - 1 == len(self.lut1)

        lut = pred[0] * self.lut0.lut
        for k, lut1 in self.lut1.foreach():
            lut += pred[k+1] * lut1.lut

        return lut

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

        print('load class model {}'.format(class_name))
        print('load lut model {}'.format(lut_name))

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

    def disable_train(self):
        self.lut0.eval()
        self.lut1.eval()
        self.classifier.eval()
        return


if __name__ == '__main__':
    unparie = False
    eval = EvalPaired(33, 2, device='cuda', unpaired=unparie)
    model_path ='/mnt/data/train.output/imagelut'

    out_path = '/mnt/data/train.output/imagelut/output'
    cls_name = 'classifier_paired_None.pth'
    lut_name = 'luts_paired_None.pth'
    img_root = '/mnt/data/fiveK'
    eval.load_model(model_path, '{},{}'.format(lut_name, cls_name))

    with open(os.path.join(img_root, 'test.txt'), mode='r') as h:
        lines = h.readlines()

    for line in lines:
        name = line.strip('\n')
        img_path = os.path.join(img_root, 'input/JPG/480p', '{}.jpg'.format(name))
        expert_path = os.path.join(img_root, 'expertC/JPG/480p', '{}.jpg'.format(name))

        img = Image.open(img_path)
        ndarr = eval.inference(img)

        w, h = img.size
        im = Image.new(img.mode, size=(w*3, h))
        im.paste(img, (0, 0))
        im.paste(Image.fromarray(ndarr), (w, 0))
        im.paste(Image.open(expert_path), (w*2, 0))

        im.save('{}/{}.jpg'.format(out_path, name), quality=95)

