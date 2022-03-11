import os
import numpy as np
from torch.utils.data import Dataset
import dataloader.torchvision_x_functional as TF_x
from dataloader.build import DATASET_ARCH_REGISTRY
import cv2
from tonemap.tone_map import run_bf_tone_map


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetTest(Dataset):
    def __init__(self, cfg, mode=""):
        root = cfg.DATALOADER.DATA_PATH

        self.test_input_files = [os.path.join(root, name) for name in os.listdir(root)]

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]
        return

    def __getitem__(self, index):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file)[1]

        img_input = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)

        img_input = TF_x.to_tensor(img_input)

        return {"A_input": img_input, "input_name": img_name}

    def __len__(self):
        return len(self.test_input_files)

    def get_item(self, index, skin_name, special_name=None, img_format='jpg'):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file)[1]
        if img_name.endswith('tif') and img_format != 'tif':
            img_name = '{}.{}'.format(img_name[:img_name.rfind('.tif')], img_format)

        if special_name is not None and img_name not in special_name:
            return {"A_input": None, "input_name": img_name}

        if img_name in skin_name:
            return {"A_input": None, "input_name": img_name}

        img_input = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)

        # down_sample_factor = 8
        # w, h, c = img_input.shape
        # w = (w // down_sample_factor * down_sample_factor)
        # h = (h // down_sample_factor * down_sample_factor)
        # img_input = img_input[:w, :h, :]
        #
        # final_img = run_bf_tone_map(img_input, is_rgb=True, gamma=0.6, down_sample_factor=down_sample_factor)
        # img_input = (final_img * 255.0).astype(np.uint8)

        img_input = TF_x.to_tensor(img_input)
        return {"A_input": img_input, "input_name": img_name}
