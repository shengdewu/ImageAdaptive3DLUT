import torch


class DepthWiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(DepthWiseSeparableConv2d, self).__init__()

        self.depth = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.wise = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        return

    def forward(self, x):
        x = self.depth(x)
        return self.wise(x)
