import torch
from .resnet import create_resnet
from models.functional import discriminator_block
from models.classifier.build import CLASSIFIER_ARCH_REGISTRY
from models.functional import weights_init_normal
import logging
from engine.checkpoint.functional import load_model_state_dict
import torch.nn.functional as torch_func
import torchvision.models.mobilenet
import torchvision.transforms.functional as ttf


__all__ = [
    'Classifier',
    'ClassifierUnpaired',
    'ClassifierResnet',
    'ClassifierResnetSoftMax',
    'MobileNet'
]


class ConvBnReLu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn=True):
        super(ConvBnReLu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = None
        if bn:
            self.bn = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.LeakyReLU()
        return

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)


@CLASSIFIER_ARCH_REGISTRY.register()
class Classifier(torch.nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Upsample(size=(256, 256), mode='bilinear'),
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.InstanceNorm2d(16, affine=True),
            ConvBnReLu(16, 32),
            ConvBnReLu(32, 64),
            ConvBnReLu(64, 128),
            ConvBnReLu(128, 128, bn=False),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(128, cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1, 8, padding=0),
        )

        self.to(cfg.MODEL.DEVICE)

        self.down_factor = cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1)
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        return

    def init_normal_classifier(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.InstanceNorm2d):
                if m.affine:
                    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)

        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)
        return

    def forward(self, img_input):
        if self.down_factor > 1:
            return self.model(torch_func.interpolate(img_input, scale_factor=1/self.down_factor, mode='bilinear'))
        else:
            return self.model(img_input)


# @CLASSIFIER_ARCH_REGISTRY.register()
# class Classifier(torch.nn.Module):
#     def __init__(self, cfg):
#         super(Classifier, self).__init__()
#
#         self.model = torch.nn.Sequential(
#             torch.nn.Upsample(size=(256, 256), mode='bilinear'),
#             torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             torch.nn.LeakyReLU(0.2),
#             torch.nn.InstanceNorm2d(16, affine=True),
#             *discriminator_block(16, 32, normalization=True),
#             *discriminator_block(32, 64, normalization=True),
#             *discriminator_block(64, 128, normalization=True),
#             *discriminator_block(128, 128),
#             # *discriminator_block(128, 128, normalization=True),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Conv2d(128, cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1, 8, padding=0),
#         )
#         self.to(cfg.MODEL.DEVICE)
#
#         self.down_factor = cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1)
#         assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)
#
#         logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
#         return
#
#     def init_normal_classifier(self):
#         self.apply(weights_init_normal)
#         #torch.nn.init.constant_(self.model[12].bias.data, 1.0)  # last layer paper error ?
#         torch.nn.init.constant_(self.model[-1].bias.data, 1.0)
#         return
#
#     def forward(self, img_input):
#         if self.down_factor > 1:
#             return self.model(torch_func.interpolate(img_input, scale_factor=1/self.down_factor, mode='bilinear'))
#         else:
#             return self.model(img_input)


@CLASSIFIER_ARCH_REGISTRY.register()
class ClassifierUnpaired(torch.nn.Module):
    def __init__(self, cfg):
        super(ClassifierUnpaired, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Upsample(size=(256, 256), mode='bilinear'),
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128),
            torch.nn.Conv2d(128, cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1, 8, padding=0),
        )
        self.to(cfg.MODEL.DEVICE)

        self.down_factor = cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1)
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
        return

    def init_normal_classifier(self):
        self.apply(weights_init_normal)
        #torch.nn.init.constant_(self.model[12].bias.data, 1.0)  # last layer paper error?
        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)  # last layer paper error?
        return

    def forward(self, img_input):
        if self.down_factor > 1:
            return self.model(torch_func.interpolate(img_input, scale_factor=1/self.down_factor, mode='bilinear'))
        else:
            return self.model(img_input)


@CLASSIFIER_ARCH_REGISTRY.register()
class ClassifierResnet(torch.nn.Module):
    def __init__(self, cfg):
        super(ClassifierResnet, self).__init__()
        kwargs = dict()
        kwargs['num_classes'] = cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1
        if cfg.MODEL.CLASSIFIER.RESNET_NORMAL == "InstanceNorm2d":
            kwargs['norm_layer'] = torch.nn.InstanceNorm2d

        self.resnet = create_resnet(arch=cfg.MODEL.CLASSIFIER.RESNET_ARCH, **kwargs)
        self.external_init = True
        if isinstance(cfg.MODEL.CLASSIFIER.PRETRAINED_PATH, str) and cfg.MODEL.CLASSIFIER.PRETRAINED_PATH != '':
            state_dict = torch.load(cfg.MODEL.CLASSIFIER.PRETRAINED_PATH, 'cpu' if cfg.MODEL.DEVICE == 'cpu' else 'cuda')
            load_model_state_dict(self.resnet, state_dict, cfg.OUTPUT_LOG_NAME)
            logging.getLogger(cfg.OUTPUT_LOG_NAME).info('load model {} from resnet'.format(cfg.MODEL.CLASSIFIER.PRETRAINED_PATH))
            self.external_init = False

        self.to(cfg.MODEL.DEVICE)

        self.down_factor = cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1)
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
        return

    def init_normal_classifier(self):
        if self.external_init:
            self.apply(weights_init_normal)
        torch.nn.init.constant_(self.resnet.fc.bias.data, 1.0)  # last layer
        return

    def forward(self, x):
        if self.down_factor > 1:
            return self.resnet(torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear'))
        else:
            return self.resnet(x)


@CLASSIFIER_ARCH_REGISTRY.register()
class ClassifierResnetSoftMax(ClassifierResnet):
    def __init__(self, cfg):
        super(ClassifierResnetSoftMax, self).__init__(cfg)
        self.softmax = torch.nn.Softmax(dim=1)
        return

    def forward(self, x):
        c = self.resnet(x)
        return self.softmax(c)


@CLASSIFIER_ARCH_REGISTRY.register()
class MobileNet(torch.nn.Module):
    def __init__(self, cfg):
        super(MobileNet, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(num_classes=cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1)

        self.rough_size = cfg.MODEL.CLASSIFIER.get('ROUGH_SIZE', None)

        self.blur_size = cfg.MODEL.CLASSIFIER.get('BLUR_SIZE', None)

        if self.rough_size is not None:
            self.down_factor = 1
        else:
            self.down_factor = cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1)

        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)
        self.to(cfg.MODEL.DEVICE)
        return

    def forward(self, x):
        if self.down_factor > 1:
            if self.blur_size is not None:
                x = ttf.gaussian_blur(x, kernel_size=self.blur_size)
            return self.backbone(torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear'))
        else:
            if self.rough_size is not None:
                x = ttf.resize(x, (self.rough_size, self.rough_size), interpolation=ttf.InterpolationMode.BILINEAR)
            if self.blur_size is not None:
                x = ttf.gaussian_blur(x, kernel_size=self.blur_size)
            return self.backbone(x)

    def init_normal_classifier(self):
        pass
