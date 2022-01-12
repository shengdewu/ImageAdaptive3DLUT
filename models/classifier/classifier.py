import torch
from .resnet import create_resnet
from models.functional import discriminator_block
from models.classifier.build import CLASSIFIER_ARCH_REGISTRY
from models.functional import weights_init_normal
import logging
from engine.checkpoint.functional import load_model_state_dict

__all__ = [
    'Classifier',
    'ClassifierUnpaired',
    'ClassifierResnet',
    'ClassifierResnetSoftMax'
]


@CLASSIFIER_ARCH_REGISTRY.register()
class Classifier(torch.nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Upsample(size=(256, 256), mode='bilinear'),
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(128, cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1, 8, padding=0),
        )
        self.to(cfg.MODEL.DEVICE)
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
        return

    def init_normal_classifier(self):
        self.apply(weights_init_normal)
        #torch.nn.init.constant_(self.model[12].bias.data, 1.0)  # last layer paper error ?
        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)
        return

    def forward(self, img_input):
        return self.model(img_input)


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
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
        return

    def init_normal_classifier(self):
        self.apply(weights_init_normal)
        #torch.nn.init.constant_(self.model[12].bias.data, 1.0)  # last layer paper error?
        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)  # last layer paper error?
        return

    def forward(self, img_input):
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
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}/{} as classifier'.format(cfg.MODEL.CLASSIFIER.ARCH, self.__class__))
        return

    def init_normal_classifier(self):
        if self.external_init:
            self.apply(weights_init_normal)
        torch.nn.init.constant_(self.resnet.fc.bias.data, 1.0)  # last layer
        return

    def forward(self, x):
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
