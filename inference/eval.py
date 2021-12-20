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
from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import os
import tqdm
import dataloader.torchvision_x_functional as TF_x


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

            pos = input_name.lower().rfind('.{}'.format(format))
            Inference.save_lut(combine_lut.detach().cpu().numpy(), '{}/{}.lut.{}'.format(self.output, input_name[:pos], input_name[pos + 1:]), self.unnormalizing_value)

            _, fake_B = self.triliear(combine_lut, real_A)
            img_sample = torch.cat((real_A, fake_B, real_B), -1)
            Inference.save_image(img_sample, '{}/{}'.format(self.output, input_name), unnormalizing_value=self.unnormalizing_value, nrow=1, normalize=False)

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
        return lut2d


if __name__ == '__main__':
    # args = default_argument_parser().parse_args()
    # cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    #
    # eval = Inference(cfg)
    # eval.resume_or_load()
    # root_path = '/mnt/data/data.set/xintu.data/xt.image.enhancement.540/rt_test'
    # out_path = '/home/shengdewu/data_shadow/train.output/imagelut.test'
    # for name in os.listdir(root_path):
    #     img_input = cv2.cvtColor(cv2.imread(os.path.join(root_path, name), -1), cv2.COLOR_BGR2RGB)
    #     img_input = TF_x.to_tensor(img_input).unsqueeze(0).to(cfg.MODEL.DEVICE)
    #     fake_B = eval.generator(img_input)
    #     img_sample = torch.cat((img_input, fake_B), -1)
    #     eval.save_image(img_sample, '{}/{}'.format(out_path, name), unnormalizing_value=eval.unnormalizing_value, nrow=1, normalize=False)
    #
    # eval.loop()

    root_path = '/home/shengdewu/data_shadow/train.output/imagelut.test'
    out_path = '/home/shengdewu/data_shadow/train.output/imagelut.test/base'
    os.makedirs(out_path, exist_ok=True)

    base_path = os.path.join(root_path, 'base.normal')
    compare_path_18 = os.path.join(root_path, 'base.warm')
    # compare_path_12 = os.path.join(root_path, 'c12')

    base_names = [name for name in os.listdir(base_path) if name.find('lut') == -1]
    for name in tqdm.tqdm(base_names):
        compare_name_18 = os.path.join(compare_path_18, name)
        # compare_name_12 = os.path.join(compare_path_12, name)

        if not os.path.exists(compare_name_18):# or not os.path.exists(compare_name_12):
            continue
        base_img = cv2.imread(os.path.join(base_path, name), cv2.IMREAD_UNCHANGED)
        compare_img_18 = cv2.imread(compare_name_18, cv2.IMREAD_UNCHANGED)
        # compare_img_12 = cv2.imread(compare_name_12, cv2.IMREAD_UNCHANGED)
        h, w, c = base_img.shape
        h18, w18, _ = compare_img_18.shape
        # h12, w12, _ = compare_img_12.shape
        # concat = np.zeros(shape=(max(h18, h12, h)*3, max(w, w18, w12), c), dtype=base_img.dtype)
        concat = np.zeros(shape=(max(h18, h) * 2, max(w, w18), c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        concat[h:h + h18, :w18, :] = compare_img_18
        # concat[h + h18:h + h18 + h12, :w12, :] = compare_img_12
        cv2.imwrite(os.path.join(out_path, name), concat)


