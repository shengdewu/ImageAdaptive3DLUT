import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = list()
    layers.append(torch.nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1))
    layers.append(torch.nn.LeakyReLU(0.2))
    if normalization:
        layers.append(torch.nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(torch.nn.BatchNorm2d(out_filters))

    return layers


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if m.affine:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D: torch.nn.Module, real_samples, fake_samples, grad_outputs_shape, device='cuda'):

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(grad_outputs_shape).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
