import torch
from models.functional import discriminator_block


class Classifier(torch.nn.Module):
    def __init__(self, in_channels=3, device='cuda', class_num=3):
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
            torch.nn.Conv2d(128, class_num, 8, padding=0),
        )
        self.to(device)
        return

    def forward(self, img_input):
        return self.model(img_input)


class Classifier_unpaired(torch.nn.Module):
    def __init__(self, in_channels=3, device='cuda', class_num=3):
        super(Classifier_unpaired, self).__init__()

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
            torch.nn.Conv2d(128, class_num, 8, padding=0),
        )
        self.to(device)
        return

    def forward(self, img_input):
        return self.model(img_input)
