import torch


def _default_weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if m.affine:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


def _kaming_weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if m.affine:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


def select_weights_init(wtype):
    if wtype == 'kaiming':
        return _kaming_weights_init_normal
    else:
        return _default_weights_init_normal
