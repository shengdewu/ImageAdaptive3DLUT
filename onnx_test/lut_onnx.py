import torch
from engine.checkpoint.functional import get_model_state_dict, load_model_state_dict
from engine.checkpoint.checkpoint_state_dict import CheckPointStateDict
import numpy as np
import onnx
import onnxruntime
import torchvision
import os
import cv2
import time
from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel


def generate_identity_lut(dim):
    lut3d = np.zeros((3, dim, dim, dim), dtype=np.float32)
    step = 1.0 / float(dim - 1)
    for b in range(dim):
        for g in range(dim):
            for r in range(dim):
                lut3d[0, b, g, r] = step * r
                lut3d[1, b, g, r] = step * g
                lut3d[2, b, g, r] = step * b
    return lut3d


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


class Classifier(torch.nn.Module):
    def __init__(self, nums):
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
            torch.nn.Conv2d(128, nums + 1, 8, padding=0),
        )
        return

    def forward(self, img_input):
        return self.model(img_input)


class MobileNet(torch.nn.Module):
    def __init__(self, nums):
        super(MobileNet, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(num_classes=nums + 1)
        return

    def forward(self, x):
        return self.backbone(x)


class LutModel(torch.nn.Module):
    def __init__(self, cfg):
        super(LutModel, self).__init__()

        dim = cfg.MODEL.LUT.DIMS
        supplement_nums = cfg.MODEL.LUT.SUPPLEMENT_NUMS

        identity_buffer = generate_identity_lut(dim)
        self.lut0 = torch.nn.Parameter(torch.from_numpy(identity_buffer)).requires_grad_(False)

        self.supplement_lut = dict()
        for i in range(supplement_nums):
            self.supplement_lut[i] = torch.nn.Parameter(torch.zeros(3, dim, dim, dim, dtype=torch.float)).requires_grad_(False)

        self.classifier = MobileNet(cfg.MODEL.LUT.SUPPLEMENT_NUMS)

        self.eval()
        return

    def forward(self, img):
        assert img.shape[0] == 1

        with torch.no_grad():
            img_cls = self.classifier(img).squeeze()

        assert img_cls.shape[0] - 1 == len(self.supplement_lut)

        combine_lut = img_cls[0] * self.lut0
        for i, lut in self.supplement_lut.items():
            combine_lut += img_cls[i + 1] * lut
        return torch.clip(combine_lut, 0.0, 1.0)


def load_state_dict(module: LutModel, state_dict: dict, log_name="image.lut"):
    assert len(state_dict['lut']) == 1 + len(module.supplement_lut)

    module.lut0.copy_(state_dict['lut'][0]['_lut'])
    keys = module.supplement_lut.keys()
    for key in keys:
        module.supplement_lut[key].copy_(state_dict['lut'][key+1]['_lut'])

    load_model_state_dict(module.classifier, state_dict['cls'], log_name=log_name)
    return


def get_state_dict(module: LutModel):
    state_dict = dict()
    state_dict['lut'] = {0: module.lut0}
    for i, lut in module.supplement_lut.items():
        state_dict['lut'].update({i+1: lut})
    state_dict['cls'] = get_model_state_dict(module.classifier)
    return state_dict


def to_onnx(cfg, onnx_name, input_size, device='cpu', log_name=''):
    model_state_dict, _ = CheckPointStateDict(save_dir='', save_to_disk=False).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    lut_model = LutModel(cfg)
    load_state_dict(lut_model, model_state_dict, log_name=log_name)

    torch.save(get_state_dict(lut_model), './lut.pth')

    torch.onnx.export(lut_model,
                      torch.zeros(size=input_size, device=device, dtype=torch.float32),
                      onnx_name,
                      # export_params=False,
                      dynamic_axes={'input_img': {2: 'h', 3: 'w'}},
                      input_names=['input_img'],
                      output_names=['out_lut'],
                      opset_version=11)

    model = onnx.load(onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    return


def load_onnx(onnx_name):
    model = onnx.load(onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    ort_session = onnxruntime.InferenceSession(onnx_name,
                                               providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    return ort_session


def normalized(img):
    '''convert numpy.ndarray to torch tensor. \n
        if the image is uint8 , it will be divided by 255;\n
        if the image is uint16 , it will be divided by 65535;\n
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n

    Arguments:
        img {numpy.ndarray} -- image to be converted to tensor.
    '''
    if not isinstance(img, np.ndarray) and (img.ndim in {2, 3}):
        raise TypeError('data should be numpy ndarray. but got {}'.format(type(img)))

    if img.ndim == 2:
        img = img[:, :, None]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535
    elif img.dtype in [np.float32, np.float64]:
        img = img.astype(np.float32) / 1
    else:
        raise TypeError('{} is not support'.format(img.dtype))

    return img


def save_image(tensor, fp, unnormalizing_value=255, **kwargs):
    fmt = np.uint8 if unnormalizing_value == 255 else np.uint16
    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
    ndarr = grid.mul(unnormalizing_value).add_(0.5).clamp_(0, unnormalizing_value).permute(1, 2, 0).to('cpu').numpy().astype(fmt)
    cv2.imwrite(fp, ndarr[:, :, ::-1])
    return


def onnx_run(in_path, out_path, ort_session, down_factor=1, ref_size=None):

    os.makedirs(out_path, exist_ok=True)
    cal_time = list()
    trilinear = TrilinearInterpolationModel()

    for name in os.listdir(in_path):
        if name.lower().endswith('tif'):
            img_name = name.replace('tif', 'jpg')
        else:
            img_name = name

        if os.path.exists(os.path.join(out_path, img_name)):
            continue
        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(in_path, name), -1), cv2.COLOR_BGR2RGB)
        img_input = normalized(img_rgb)

        if ref_size is not None:
            h, w, c = img_input.shape
            scale = ref_size * 1.0 / max(h, w)
            new_h = int(h * scale + 0.5)
            new_w = int(w * scale + 0.5)
            img_input = cv2.resize(img_input, (new_w, new_h), cv2.INTER_CUBIC)
        else:
            if down_factor > 1 and down_factor % 2 == 0:
                h, w, c = img_input.shape
                h = (h // down_factor) * down_factor
                w = (w // down_factor) * down_factor
                img_rgb = img_rgb[:h, :w, :]
                img_input = cv2.resize(img_input, (img_input.shape[1] // down_factor, img_input.shape[0] // down_factor), cv2.INTER_CUBIC)

        stime = time.time()
        outputs = ort_session.run(None, {'input_img': img_input.transpose((2, 0, 1))[np.newaxis, :]})
        cal_time.append(time.time() - stime)

        lut = torch.from_numpy(outputs[0])

        # save_lut = np.zeros((64, 64, 3))
        # for x_cell in range(4):
        #     for y_cell in range(4):
        #         for g in range(16):
        #             for r in range(16):
        #                 b = x_cell + y_cell * 4
        #                 x = r + x_cell * 16
        #                 y = g + y_cell * 16
        #                 save_lut[y, x, 2] = lut[0, b, g, r]
        #                 save_lut[y, x, 1] = lut[1, b, g, r]
        #                 save_lut[y, x, 0] = lut[2, b, g, r]

        # cv2.imwrite(os.path.join(out_path, '{}.jpg'.format(img_name.replace('jpg', 'lut'))), (save_lut*255).astype(np.uint8))

        real_a = torch.from_numpy(normalized(img_rgb).transpose((2, 0, 1))).unsqueeze(0)
        _, enhance_img = trilinear(lut, real_a)

        lut_gain = torch.mean(enhance_img[:, :, ::10, ::10]) / torch.mean(real_a[:, :, ::10, ::10])
        w = 32 * ((real_a - 0.5) ** 6)
        if lut_gain > 1:
            w[real_a > 0.5] = 0
        else:
            w[real_a < 0.5] = 0
        enhance_img = (1 - w) * enhance_img + w * real_a

        save_image(enhance_img, os.path.join(out_path, img_name), nrow=1, normalize=False)

