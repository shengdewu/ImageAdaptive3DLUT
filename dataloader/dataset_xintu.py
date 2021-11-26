import os
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import dataloader.torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF
from dataloader import DATASET_ARCH_REGISTRY


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetXinTu(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for name in set1_input_files:
            name = name.split(',')
            assert len(name) == 2
            a_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            a_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(a_input) or not os.path.exists(a_expert):
                continue
            self.set1_input_files.append(a_input)
            self.set1_expert_files.append(a_expert)

        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for name in set2_input_files:
            name = name.split(',')
            assert len(name) == 2
            b_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            b_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(b_input) or not os.path.exists(b_expert):
                continue
            self.set2_input_files.append(b_input)
            self.set2_expert_files.append(b_expert)

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.test_input_files = list()
        self.test_expert_files = list()
        for name in test_input_files:
            name = name.split(',')
            assert len(name) == 2
            t_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            t_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(t_input) or not os.path.exists(t_expert):
                continue
            self.test_input_files.append(t_input)
            self.test_expert_files.append(t_expert)

        self.set1_input_files = self.set1_input_files + self.set2_input_files
        self.set1_expert_files = self.set1_expert_files + self.set2_expert_files

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index])[-1]
            img_input = cv2.imread(self.set1_input_files[index], -1)
            img_exptC = cv2.imread(self.set1_expert_files[index], -1)
        else:
            img_name = os.path.split(self.test_input_files[index])[-1]
            img_input = cv2.imread(self.test_input_files[index], cv2.IMREAD_COLOR)
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_exptC = cv2.imread(self.test_expert_files[index], cv2.IMREAD_COLOR)
            img_exptC = cv2.cvtColor(img_exptC, cv2.COLOR_BGR2RGB)

        h1, w1 = img_input.shape[0: 2]
        h2, w2 = img_exptC.shape[0: 2]
        if h1 != h2 or w1 != w2:
            h = max(h1, h2)
            w = max(w1, w2)
            if h != h1 or w != w1:
                img_input = cv2.resize(img_input, dsize=(w, h))
            if h != h2 or w != w2:
                img_exptC = cv2.resize(img_exptC, dsize=(w, h))

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_input.shape[1], img_input.shape[0]
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            i, j, h, w = TF_x.get_crop_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF_x.crop(img_exptC, i, j, h, w)

            # img_input = TF_x.resized_crop(img_input, i, j, h, w, (320,320))
            # img_exptC = TF_x.resized_crop(img_exptC, i, j, h, w, (320,320))

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF_x.hflip(img_exptC)

            a = np.random.uniform(0.8, 1.2)
            img_input = TF_x.adjust_brightness(img_input, a)

            img_input = TF_x.to_tensor(img_input)
            img_exptC = TF_x.to_tensor(img_exptC)
        else:
            img_input = TF.to_tensor(img_input)
            img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        else:
            return len(self.test_input_files)


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetXinTuUnpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for name in set1_input_files:
            name = name.split(',')
            assert len(name) == 2
            a_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            a_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(a_input) or not os.path.exists(a_expert):
                continue
            self.set1_input_files.append(a_input)
            self.set1_expert_files.append(a_expert)

        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for name in set2_input_files:
            name = name.split(',')
            assert len(name) == 2
            b_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            b_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(b_input) or not os.path.exists(b_expert):
                continue
            self.set2_input_files.append(b_input)
            self.set2_expert_files.append(b_expert)

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        self.test_input_files = list()
        self.test_expert_files = list()
        for name in test_input_files:
            name = name.split(',')
            assert len(name) == 2
            t_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            t_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(t_input) or not os.path.exists(t_expert):
                continue
            self.test_input_files.append(t_input)
            self.test_expert_files.append(t_expert)
        return

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index])[-1]
            img_input = cv2.imread(self.set1_input_files[index], -1)
            img_exptC = cv2.imread(self.set1_expert_files[index], -1)
            seed = random.randint(1, len(self.set2_expert_files))
            img2 = cv2.imread(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)], -1)

        else:
            img_name = os.path.split(self.test_input_files[index])[-1]
            img_input = cv2.imread(self.test_input_files[index], cv2.IMREAD_COLOR)
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_exptC = cv2.imread(self.test_expert_files[index], cv2.IMREAD_COLOR)
            img_exptC = cv2.cvtColor(img_exptC, cv2.COLOR_BGR2RGB)
            img2 = img_exptC

        h1, w1 = img_input.shape[0: 2]
        h2, w2 = img_exptC.shape[0: 2]
        h3, w3 = img2.shape[0: 2]
        if h1 != h2 or w1 != w2 or h1 != h3 or w1 != w3:
            h = max(h1, h2, h3)
            w = max(w1, w2, w3)
            if h != h1 or w != w1:
                img_input = cv2.resize(img_input, dsize=(w, h))
            if h != h2 or w != w2:
                img_exptC = cv2.resize(img_exptC, dsize=(w, h))
            if h != h3 or w != w3:
                img2 = cv2.resize(img2, dsize=(w, h))

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_input.shape[1], img_input.shape[0]
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            W2, H2 = img2.shape[1], img2.shape[0]
            crop_h = min(crop_h, H2)
            crop_w = min(crop_w, W2)
            i, j, h, w = TF_x.get_crop_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF_x.crop(img_exptC, i, j, h, w)
            i, j, h, w = TF_x.get_crop_params(img2, output_size=(crop_h, crop_w))
            img2 = TF_x.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF_x.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF_x.hflip(img2)

            # if np.random.random() > 0.5:
            #    img_input = TF_x.vflip(img_input)
            #    img_exptC = TF_x.vflip(img_exptC)
            #    img2 = TF_x.vflip(img2)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

            img_input = TF_x.to_tensor(img_input)
            img_exptC = TF_x.to_tensor(img_exptC)
            img2 = TF_x.to_tensor(img2)
        else:
            img_input = TF.to_tensor(img_input)
            img_exptC = TF.to_tensor(img_exptC)
            img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        else:
            return len(self.test_input_files)


if __name__ == '__main__':

    with open('/mnt/data/data.set/fiveK.data/train_input.txt') as tt:
        tinput = tt.readlines()
    with open('/mnt/data/data.set/fiveK.data/train_label.txt') as lt:
        tlabel = lt.readlines()

    tinput = sorted(tinput)
    tlabel = sorted(tlabel)

    for i in range(len(tinput)):
        seed = random.randint(1, len(tlabel))
        si = (i + seed) % len(tlabel)
        print(si)

    with open('gt.txt') as rgt:
        gts = [r.strip('\n') for r in rgt.readlines()]
    with open('input.txt') as igt:
        inputs = [r.strip('\n') for r in igt.readlines()]

    select_name = list()
    for gt in gts:
        if gt in inputs:
            zero = '{}_0{}'.format(gt[:-4], gt[-4:])
            one = '{}_1{}'.format(gt[:-4], gt[-4:])
            two = '{}_2{}'.format(gt[:-4], gt[-4:])
            if zero not in inputs or one not in inputs or two not in inputs:
                print(gt)
                continue
            select_name.append((gt, gt))
            select_name.append((gt, zero))
            select_name.append((gt, one))
            select_name.append((gt, two))

    index = [i for i in range(len(select_name))]
    test_name_index = random.sample(index, int(0.1*len(select_name)))

    assert len(test_name_index) == len(set(test_name_index)), 'the test name index duplication'

    test_name = list()
    train_name = list()
    for i in index:
        if i in test_name_index:
            test_name.append(select_name[i])
        else:
            train_name.append(select_name[i])

    with open('test.txt', mode='w') as t:
        t.write('input,gt\n')
        for name in test_name:
            assert len(name) == 2
            t.write('{},{}\n'.format(name[1], name[0]))

    index = [i for i in range(len(train_name))]
    select_index = random.sample(index, int(0.5*len(train_name)))

    with open('train_input.txt', mode='w') as ti:
        with open('train_label.txt', mode='w') as tl:
            ti.write('input,gt\n')
            tl.write('input,gt\n')
            for i in index:
                name = train_name[i]
                assert len(name) == 2
                if i in select_index:
                    ti.write('{},{}\n'.format(name[1], name[0]))
                else:
                    tl.write('{},{}\n'.format(name[1], name[0]))