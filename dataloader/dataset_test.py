import os
import numpy as np
from torch.utils.data import Dataset
import dataloader.torchvision_x_functional as TF_x
from dataloader.build import DATASET_ARCH_REGISTRY
import cv2


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetTest(Dataset):
    def __init__(self, cfg, mode=""):
        root = cfg.DATALOADER.DATA_PATH
        txt_name = cfg.DATALOADER.XT_TEST_TXT

        file = open(os.path.join(root, txt_name), 'r')
        self.test_input_files = [os.path.join(root, name.strip('\n')) for name in file.readlines()]

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]
        return

    def __getitem__(self, index):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file)[-1]
        img_input = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)

        img_input = TF_x.to_tensor(img_input)

        return {"A_input": img_input, "input_name": img_name}

    def __len__(self):
        return len(self.test_input_files)
