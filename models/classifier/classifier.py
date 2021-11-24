import torch
from .resnet import create_resnet
from models.functional import discriminator_block
from models.classifier import CLASSIFIER_ARCH_REGISTRY
from models.functional import weights_init_normal_classifier


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
        return

    def init_normal_classifier(self):
        self.apply(weights_init_normal_classifier)
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
        return

    def init_normal_classifier(self):
        self.apply(weights_init_normal_classifier)
        #torch.nn.init.constant_(self.model[12].bias.data, 1.0)  # last layer paper error?
        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)  # last layer paper error?
        return

    def forward(self, img_input):
        return self.model(img_input)


@CLASSIFIER_ARCH_REGISTRY.register()
class ClassifierResnet(torch.nn.Module):
    def __init__(self, cfg):
        super(ClassifierResnet, self).__init__()
        self.resnet, self.init = create_resnet(num_classes=cfg.MODEL.LUT.SUPPLEMENT_NUMS + 1, device=cfg.MODEL.DEVICE, arch=cfg.MODEL.CLASSIFIER.RESNET_ARCH, model_path=cfg.MODEL.CLASSIFIER.PRETRAINED_PATH)
        self.to(cfg.MODEL.DEVICE)
        return

    def init_normal_classifier(self):
        if self.init:
            self.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(self.resnet.fc.bias.data, 1.0)  # last layer
        return

    def forward(self, x):
        return self.resnet(x)
