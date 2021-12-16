import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class PatchDiscriminator(torch.nn.Module):
    def __init__(self, in_channels=3, device='cuda'):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )
        self.to(device)
        return

    def forward(self, img):
        return self.model(img)
