import torch
import torchvision.models
import logging


arch_block = {
    'resnet18': (torchvision.models.resnet18, torchvision.models.resnet.BasicBlock),
    'resnet34': (torchvision.models.resnet34, torchvision.models.resnet.BasicBlock),
    'resnet50': (torchvision.models.resnet50, torchvision.models.resnet.Bottleneck),
    'resnet101': (torchvision.models.resnet101, torchvision.models.resnet.Bottleneck),
    'resnet152': (torchvision.models.resnet152, torchvision.models.resnet.Bottleneck),
    'resnext50_32x4d': (torchvision.models.resnext50_32x4d, torchvision.models.resnet.Bottleneck),
    'resnext101_32x8d': (torchvision.models.resnext101_32x8d, torchvision.models.resnet.Bottleneck),
    'wide_resnet50_2': (torchvision.models.wide_resnet50_2, torchvision.models.resnet.Bottleneck),
    'wide_resnet101_2': (torchvision.models.wide_resnet101_2, torchvision.models.resnet.Bottleneck),
}


def create_resnet(num_classes, arch='resnet152', device='cuda', model_path='', default_log_name='', **kwargs):
    external_init = True
    resnet = arch_block[arch][0](pretrained=False, **kwargs)
    if isinstance(model_path, str) and model_path != '' and issubclass(kwargs.get('norm_layer', torch.nn.BatchNorm2d), torch.nn.BatchNorm2d):
        state_dict = torch.load(model_path, 'cpu' if device == 'cpu' else 'cuda')
        resnet.load_state_dict(state_dict)
        logging.getLogger(default_log_name).info('load model {} from resnet'.format(model_path))
        external_init = False
    resnet.fc = torch.nn.Linear(512 * arch_block[arch][1].expansion, num_classes)
    return resnet, external_init
