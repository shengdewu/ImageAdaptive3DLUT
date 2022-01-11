import torch
import torchvision


class PerceptualLoss(torch.nn.Module):
    """
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace=True)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace=True)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    """
    def __init__(self, layer, device='cuda', path=''):
        super(PerceptualLoss, self).__init__()
        self.layer = layer

        self.vgg = torchvision.models.vgg16(pretrained=False)
        map_location = None if device == 'cuda' else 'cpu'
        state_dict = torch.load(path, map_location=map_location)
        self.vgg.load_state_dict(state_dict)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.eval()
        self.to(device)
        return

    def fetch(self, x, layer):
        fea = None
        for i in range(len(self.vgg.features)):
            x = self.vgg.features[i](x)
            if i == layer:
                fea = x
                break

        return fea

    def forward(self, x, y):

        x_fea = self.fetch(x, self.layer)
        y_fea = self.fetch(y, self.layer)

        return torch.mean((x_fea - y_fea) ** 2)


if __name__ == '__main__':
    import os
    import random
    import cv2
    import dataloader.torchvision_x_functional as TF_x
    import torchvision.transforms.functional as TF
    import shutil
    import numpy as np

    def featch(img, l, device):
        fea = loss.fetch(TF_x.to_tensor(img).to(device).unsqueeze(0), l)
        fea = torch.mean((fea * 65535), dim=1).cpu().numpy()[0].astype(np.uint16)
        return fea

    device = 'cuda'
    img_root = '/mnt/data/data.set/xintu.data/xt.image.enhancement.540/gt_16bit_540p'
    out_root = '/mnt/data/train.output/imagelut.test/test.vgg'
    file_names = os.listdir(img_root)

    loss = PerceptualLoss(17, device=device, path='/mnt/data/pretrain.model/vgg.model/pytorch/vgg16-397923af.pth')

    #4, 9, 16, 30
    layers =  [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    rlayers =   [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 23, 25, 27, 29]
    for name in random.sample(file_names, 4):
        img = cv2.imread(os.path.join(img_root, name), cv2.IMREAD_UNCHANGED)
        os.makedirs(os.path.join(out_root, name), exist_ok=True)
        shutil.copy2(os.path.join(img_root, name), os.path.join(out_root, name, name))
        for l in layers:
            fea = featch(img, l, device)
            cv2.imwrite(os.path.join(out_root, name, 'conv-{}-{}'.format(l, name)), fea)

        for l in rlayers:
            fea = featch(img, l, device)
            cv2.imwrite(os.path.join(out_root, name, 'relu-{}-{}'.format(l, name)), fea)
