import logging
from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel
from engine.checkpoint.CheckpointerStateDict import CheckpointerStateDict
from models.build import build_model
import engine.comm as comm
from engine.log.logger import setup_logger
import torch
from dataloader.dataloader import DataLoader
import torchvision
import cv2
import numpy as np
from dataloader.build import build_dataset
from models.lut.lut_abc import transfer3d_2d
import os
import tqdm


class Inference:
    criterion_pixelwise = torch.nn.MSELoss()

    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        self.model = build_model(cfg)
        self.model.disable_train()

        self.test_dataset = build_dataset(cfg, model='test')

        logging.getLogger(__name__).info('create dataset {}, load {} test data'.format(cfg.DATALOADER.DATASET, len(self.test_dataset)))

        self.unnormalizing_value = self.test_dataset.unnormalizing_value() if hasattr(self.test_dataset, 'unnormalizing_value') else 255

        self.checkpointer = CheckpointerStateDict(save_dir='', save_to_disk=False)

        self.triliear = TrilinearInterpolationModel().to(cfg.MODEL.DEVICE)
        self.device = cfg.MODEL.DEVICE
        self.model_path = cfg.MODEL.WEIGHTS
        self.output = cfg.OUTPUT_DIR
        return

    def loop(self):
        format = 'jpg' if self.unnormalizing_value == 255 else 'tif'
        skin_name = [name for name in os.listdir(self.output) if name.lower().endswith(format)]
        for index in tqdm.tqdm(range(len(self.test_dataset))):
            data = DataLoader.fromlist([self.test_dataset[index]])
            input_name = data['input_name'][0]
            if input_name in skin_name:
                continue
            real_A = data["A_input"].to(self.device)
            real_B = data["A_exptC"].to(self.device)

            combine_lut = self.model.generate_lut(real_A)
            _, fake_B = self.triliear(combine_lut, real_A)
            img_sample = torch.cat((real_A, fake_B, real_B), -1)

            Inference.save_image(img_sample, '{}/{}'.format(self.output, input_name), unnormalizing_value=self.unnormalizing_value, nrow=1, normalize=False)
            pos = input_name.lower().rfind('.{}'.format(format))
            Inference.save_lut(combine_lut.detach().cpu().numpy(), '{}/{}.lut.{}'.format(self.output, input_name[:pos], input_name[pos + 1:]), self.unnormalizing_value)

            # fake_B = torch.round(fake_B * self.unnormalizing_value)
            # real_B = torch.round(real_B * self.unnormalizing_value)
            # mse = Inference.criterion_pixelwise(fake_B, real_B)
            # mse = torch.clip(mse, 0.00000001, 4294967296.0)
            # psnr = 10.0 * math.log10(float(self.unnormalizing_value) * self.unnormalizing_value / mse.item())

        return

    def generator(self, real_A):
        combine_lut = self.model.generate_lut(real_A)
        _, fake_B = self.triliear(combine_lut, real_A)
        return fake_B

    def resume_or_load(self):
        model_state_dict, addition_state_dict = self.checkpointer.resume_or_load(self.model_path, resume=False)
        addition_state_dict.pop("iteration")
        self.model.load_state_dict(model_state_dict)
        self.model.load_addition_state_dict(addition_state_dict)
        logging.getLogger(__name__).info('load model from {}'.format(self.model_path))
        return

    @staticmethod
    @torch.no_grad()
    def save_image(tensor, fp, unnormalizing_value=255, **kwargs):
        fmt = np.uint8 if unnormalizing_value == 255 else np.uint16
        grid = torchvision.utils.make_grid(tensor, **kwargs)
        # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
        ndarr = grid.mul(unnormalizing_value).add_(0.5).clamp_(0, unnormalizing_value).permute(1, 2, 0).to('cpu').numpy().astype(fmt)
        cv2.imwrite(fp, ndarr[:, :, ::-1])
        return

    @staticmethod
    @torch.no_grad()
    def save_lut(lut2d, fp, unnormalizing_value=255):
        lut2d = Inference.transfer2cube(512, lut2d)
        fmt = np.uint8 if unnormalizing_value == 255 else np.uint16
        cv2.imwrite(fp, fmt(lut2d * unnormalizing_value))

    @staticmethod
    def transfer2cube(size, lut3d):
        n, dim1, dim2, dim3 = lut3d.shape
        assert dim1 == dim2 and dim1 == dim3
        assert size % dim1 == 0
        box = int(size / dim1)
        assert box * box == dim1, 'the box power({}, 2) must be == {}'.format(box, dim1)
        lut2d = np.zeros((size, size, 3), dtype=np.float32)
        transfer3d_2d(box, dim1, lut3d, lut2d)
        return lut2d[:, :, ::-1]


class KRResize:
    def __init__(self, short_edge_length, interp=cv2.INTER_LINEAR):
        self.short_edge_length = short_edge_length
        self.interp = interp
        return

    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        h, w, c = image.shape

        scala = self.short_edge_length * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = self.short_edge_length, w * scala
        else:
            new_h, new_w = h * scala, self.short_edge_length

        new_h, new_w = int(new_h + 0.5), int(new_w + 0.5)

        return cv2.resize(image, dsize=(new_w, new_h), interpolation=self.interp)

    def __str__(self):
        return 'RandomResize'


def compare(base_path, compare_paths, out_path, skip=False):
    os.makedirs(out_path, exist_ok=True)
    skip_name = list()
    if skip:
        skip_name = os.listdir(out_path)

    base_names = [name for name in os.listdir(base_path) if name.find('lut') == -1]
    for name in tqdm.tqdm(base_names):
        if name in skip_name:
            continue

        compare_names = [os.path.join(compare_path, name) for compare_path in compare_paths]

        is_exist = [not os.path.exists(compare_name) for compare_name in compare_names]
        if np.any(is_exist):
            continue
        base_img = cv2.imread(os.path.join(base_path, name), cv2.IMREAD_UNCHANGED)
        compare_imgs = [cv2.imread(compare_name, cv2.IMREAD_UNCHANGED) for compare_name in compare_names]
        h, w, c = base_img.shape
        shapes = [compare_img.shape for compare_img in compare_imgs]
        ch = [shape[0] for shape in shapes]
        cw = [shape[1] for shape in shapes]
        ch = sum(ch)
        cw = max(cw)
        concat = np.zeros(shape=(ch+h, max(w, cw), c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        start_h = h
        for i in range(len(compare_imgs)):
            h, w, c = shapes[i]
            assert shapes[i] == compare_imgs[i].shape
            concat[start_h:h+start_h, :w, :] = compare_imgs[i]
            start_h += h
        cv2.imwrite(os.path.join(out_path, name), concat)
    return



