import torchvision.transforms.functional as F
import engine.transforms.functional as ETF
import numpy as np
import logging
import torch


def range_float(start, end, step, exclude, base=10):
    assert start <= end
    return [i/base for i in np.arange(int(start*base), int(end*base+1), step=int(step*base)) if i != int(exclude*base)]


class Saturation:
    """
    调整图像的饱和度, 0.0 黑白图
    """
    def __init__(self, f_min=0.8, f_max=1.6):
        self.factor = range_float(f_min, f_max, step=0.05, exclude=1.0, base=100)
        return

    def __call__(self, image, factor=None):
        random_factor = np.random.choice(self.factor, 1, replace=False)[0] if factor is None else factor
        return F.adjust_saturation(image, random_factor)

    def __str__(self):
        return 'Saturation'


class Brightness:
    def __init__(self, f_min=0.6, f_max=1.1):
        self.factor = range_float(f_min, f_max, step=0.05, exclude=1.0, base=100)
        return

    def __call__(self, image, factor=None):
        random_factor = np.random.choice(self.factor, 1, replace=False)[0] if factor is None else factor
        return F.adjust_brightness(image, random_factor)

    def __str__(self):
        return 'Brightness'


class Contrast:
    """
    调整图像的对比度
    """
    def __init__(self, f_min=0.6, f_max=1.2):
        self.factor = range_float(f_min, f_max, step=0.05, exclude=1.0, base=100)
        return

    def __call__(self, image, factor=None):
        random_factor = np.random.choice(self.factor, 1, replace=False)[0] if factor is None else factor
        return F.adjust_contrast(image, random_factor)

    def __str__(self):
        return 'Contrast'


class Hue:
    def __init__(self, f_min=-0.5, f_max=0.5):
        f_min = f_min if f_min > -0.5 else -0.5
        f_max = f_max if f_max < 0.5 else 0.5
        self.factor = range_float(f_min, f_max, step=0.05, exclude=1.0, base=100)
        return

    def __call__(self, image, factor=None):
        random_factor = np.random.choice(self.factor, 1, replace=False)[0] if factor is None else factor
        return F.adjust_hue(image, random_factor)

    def __str__(self):
        return 'Hue'


class AdaptiveLight:
    def __init__(self, f_min=0.6, f_max=1.1, log_name=''):
        self.factor = range_float(f_min, f_max, step=0.05, exclude=1.0, base=100)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = np.random.choice(self.factor, 1, replace=False)[0] if factor is None else factor
        f_min = 1.0
        f_max = random_factor
        if random_factor < 1.0:
            f_min = random_factor
            f_max = 1.0
        return ETF.adjust_brightness_adaptive_inv(image, f_min, f_max)

    def __str__(self):
        return 'AdaptiveLight'


class ColorJitter:

    def __init__(self, cfg, log_name):
        self.method = list()

        if cfg.BRIGHTNESS.ENABLE:
            self.method.append(Brightness(f_min=cfg.BRIGHTNESS.MIN, f_max=cfg.BRIGHTNESS.MAX))
            logging.getLogger(log_name).info('enable {}-({},{})'.format(self.method[-1], cfg.BRIGHTNESS.MIN, cfg.BRIGHTNESS.MAX))

        if cfg.SATURATION.ENABLE:
            self.method.append(Saturation(f_min=cfg.SATURATION.MIN, f_max=cfg.SATURATION.MAX))
            logging.getLogger(log_name).info('enable {}-({},{})'.format(self.method[-1], cfg.SATURATION.MIN, cfg.SATURATION.MAX))

        if cfg.CONTRAST.ENABLE:
            self.method.append(Contrast(f_min=cfg.CONTRAST.MIN, f_max=cfg.CONTRAST.MAX))
            logging.getLogger(log_name).info('enable {}-({},{})'.format(self.method[-1], cfg.CONTRAST.MIN, cfg.CONTRAST.MAX))

        if cfg.ADAPTIVE_LIGHT.ENABLED:
            self.method.append(AdaptiveLight(f_min=cfg.ADAPTIVE_LIGHT.MIN,
                                             f_max=cfg.ADAPTIVE_LIGHT.MAX,
                                             log_name=log_name))

        self.method_id = [i for i in range(len(self.method))]
        return

    def __call__(self, image, factor=None):
        """
        Args:
            image (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        method_id = np.random.choice(self.method_id, size=1, replace=False)[0]
        return self.method[method_id](image, factor=factor)

    def __str__(self):
        return 'ColorJitter'


class AdaptiveOverExposure:
    def __init__(self, f_min=0.0, f_max=0.1, f_value=2.0, log_name=''):
        self.f_min = f_min
        self.f_max = f_max
        self.f_value = f_value
        logging.getLogger(log_name).info('{}/f_min={};f_max={},f_value={}'.format(self.__class__, self.f_min, self.f_max, self.f_value))

    def __call__(self, img_input, img_expert):
        """
        img_input c, h, w belong to [0, 1.0]
        img_expert c, h, w belong to [0, 1.0]
        """
        expert_gray = float(torch.mean(0.299 * img_expert[0] + 0.587 * img_expert[1] + 0.114 * img_expert[2]))
        input_gray = float(torch.mean(0.299 * img_input[0] + 0.587 * img_input[1] + 0.114 * img_input[2]))

        diff_gray = expert_gray - input_gray

        if self.f_min <= diff_gray <= self.f_max:
            img_input = ETF.adjust_brightness_adaptive(img_input, 1.0, self.f_value)

        return img_input
