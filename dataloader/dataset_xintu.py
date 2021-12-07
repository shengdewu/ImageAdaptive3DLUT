import os
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import dataloader.torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF
from dataloader import DATASET_ARCH_REGISTRY
from PIL import Image
import torchvision.transforms as transforms


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetXinTu(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        self.skip_name = list()
        if os.path.exists(os.path.join(root, 'skip.txt')):
            file = open(os.path.join(root, 'skip.txt'), 'r')
            self.skip_name = [name.strip('\n') for name in file.readlines()]
            file.close()

        self.set1_input_files, self.set1_expert_files = ImageDatasetXinTu.search_files(root, 'train_input.txt', self.skip_name)
        self.set2_input_files, self.set2_expert_files = ImageDatasetXinTu.search_files(root, 'train_label.txt', self.skip_name)
        self.test_input_files, self.test_expert_files = ImageDatasetXinTu.search_files(root, 'test.txt', self.skip_name)

        self.set1_input_files = self.set1_input_files + self.set2_input_files
        self.set1_expert_files = self.set1_expert_files + self.set2_expert_files

    @staticmethod
    def search_files(root, txt, skip_name):
        file = open(os.path.join(root, txt), 'r')
        sorted_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        file.close()
        input_files = list()
        expert_files = list()
        for name in sorted_input_files:
            name = name.split(',')
            assert len(name) == 2
            if name[0] in skip_name or name[1] in skip_name:
                continue
            b_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            b_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(b_input) or not os.path.exists(b_expert):
                continue
            input_files.append(b_input)
            expert_files.append(b_expert)

        return input_files, expert_files

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        else:
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_input._size
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            # img_input = TF.resized_crop(img_input, i, j, h, w, (320,320))
            # img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (320,320))

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.8, 1.2)
            img_input = TF.adjust_brightness(img_input, a)

            a = np.random.uniform(0.8, 1.2)
            img_input = TF.adjust_saturation(img_input, a)

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

        self.skip_name = list()
        if os.path.exists(os.path.join(root, 'skip.txt')):
            file = open(os.path.join(root, 'skip.txt'), 'r')
            self.skip_name = [name.strip('\n') for name in file.readlines()]
            file.close()

        self.set1_input_files, self.set1_expert_files = ImageDatasetXinTuUnpaired.search_files(root, 'train_input.txt')
        self.set2_input_files, self.set2_expert_files = ImageDatasetXinTuUnpaired.search_files(root, 'train_label.txt')
        self.test_input_files, self.test_expert_files = ImageDatasetXinTuUnpaired.search_files(root, 'test.txt')

        return

    @staticmethod
    def search_files(root, txt):
        file = open(os.path.join(root, txt), 'r')
        sorted_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
        file.close()
        input_files = list()
        expert_files = list()
        for name in sorted_input_files:
            name = name.split(',')
            assert len(name) == 2
            b_input = os.path.join(root, "rt_tif_16bit_540p", name[0])
            b_expert = os.path.join(root, "gt_16bit_540p", name[1])
            if not os.path.exists(b_input) or not os.path.exists(b_expert):
                continue
            input_files.append(b_input)
            expert_files.append(b_expert)

        return input_files, expert_files

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1, len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        else:
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_input._size
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)
            W2, H2 = img2._size
            crop_h = min(crop_h, H2)
            crop_w = min(crop_w, W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            # if np.random.random() > 0.5:
            #    img_input = TF.vflip(img_input)
            #    img_exptC = TF.vflip(img_exptC)
            #    img2 = TF.vflip(img2)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF.adjust_brightness(img_input, a)

            a = np.random.uniform(0.8, 1.2)
            img_input = TF.adjust_saturation(img_input, a)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        else:
            return len(self.test_input_files)


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetXinTuTif(ImageDatasetXinTu):
    def __init__(self, root, mode="train"):
        super(ImageDatasetXinTuTif, self).__init__(root, mode)
        return

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index])[-1]
            img_input = cv2.cvtColor(cv2.imread(self.set1_input_files[index], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(self.set1_expert_files[index], -1), cv2.COLOR_BGR2RGB)
        else:
            img_name = os.path.split(self.test_input_files[index])[-1]
            img_input = cv2.cvtColor(cv2.imread(self.test_input_files[index], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(self.test_expert_files[index], -1), cv2.COLOR_BGR2RGB)

        if self.mode == "train":

            W = min(img_input.shape[1], img_exptC.shape[1])
            H = min(img_input.shape[0], img_exptC.shape[0])
            if img_input.shape[:2] != (H, W):
                img_input = img_input[0:H, 0:W, :]
            if img_exptC.shape[:2] != (H, W):
                img_exptC = img_exptC[0:H, 0:W, :]

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

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def unnormalizing_value(self):
        return 65535


@DATASET_ARCH_REGISTRY.register()
class ImageDatasetXinTuUnpairedTif(ImageDatasetXinTuUnpaired):
    def __init__(self, root, mode="train"):
        super(ImageDatasetXinTuUnpairedTif, self).__init__(root, mode)
        return

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index])[-1]
            img_input = cv2.cvtColor(cv2.imread(self.set1_input_files[index], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(self.set1_expert_files[index], -1), cv2.COLOR_BGR2RGB)
            seed = random.randint(1, len(self.set2_expert_files))
            img2 = cv2.cvtColor(cv2.imread(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)], -1), cv2.COLOR_BGR2RGB)

        else:
            img_name = os.path.split(self.test_input_files[index])[-1]
            img_input = cv2.cvtColor(cv2.imread(self.test_input_files[index], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(self.test_expert_files[index], -1), cv2.COLOR_BGR2RGB)
            img2 = img_exptC

        if self.mode == "train":
            W = min(img_input.shape[1], img_exptC.shape[1])
            H = min(img_input.shape[0], img_exptC.shape[0])
            if img_input.shape[:2] != (H, W):
                img_input = img_input[0:H, 0:W, :]
            if img_exptC.shape[:2] != (H, W):
                img_exptC = img_exptC[0:H, 0:W, :]

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

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def unnormalizing_value(self):
        return 65535

def singal_preprocess(data_path, out_path, txt):
    file = open(os.path.join(data_path, txt), 'r')
    input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()
    with open(os.path.join(out_path, txt), mode='w') as w:
        w.write('input,gt\n')

        for name in input_files:
            name = name.split(',')
            assert len(name) == 2
            a_input = os.path.join(data_path, "rt_tif_16bit_540p", name[0])
            a_expert = os.path.join(data_path, "gt_16bit_540p", name[1])
            if not os.path.exists(a_input) or not os.path.exists(a_expert):
                continue

            a_input_img = Image.open(a_input)
            a_expert_img = Image.open(a_expert)

            W, H = a_input_img.size
            W2, H2 = a_expert_img.size

            if abs(W - W2) > 2 or abs(H - H2) > 2:
                continue

            min_w = max(W, W2)
            min_h = max(H, H2)

            a_input_rgb = a_input_img.convert('RGB')
            if a_input_rgb.size != (min_w, min_h):
                a_input_rgb = a_input_rgb.resize((min_w, min_h), Image.BILINEAR)
            a_input_rgb.save(os.path.join(out_path, "rt_tif_16bit_540p", '{}.jpg'.format(name[0][0:name[0].rfind('.tif')])))

            a_expert_rgb = a_expert_img.convert('RGB')
            if a_expert_rgb.size != (min_w, min_h):
                a_expert_rgb = a_expert_rgb.resize((min_w, min_h), Image.BILINEAR)
            a_expert_rgb.save(os.path.join(out_path, "gt_16bit_540p", '{}.jpg'.format(name[1][0:name[1].rfind('.tif')])))

            w.write('{},{}\n'.format('{}.jpg'.format(name[0][0:name[0].rfind('.tif')]), '{}.jpg'.format(name[1][0:name[1].rfind('.tif')])))


def preprocess(data_path, out_path):
    os.makedirs(os.path.join(out_path, 'rt_tif_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'gt_16bit_540p'), exist_ok=True)

    singal_preprocess(data_path, out_path, 'train_input.txt')
    singal_preprocess(data_path, out_path, 'train_label.txt')
    singal_preprocess(data_path, out_path, 'test.txt')


def singal_match(data_path, txt):
    file = open(os.path.join(data_path, txt), 'r')
    input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()

    skip_names = list()
    for name in input_files:
        name = name.split(',')
        assert len(name) == 2
        a_input = os.path.join(data_path, "rt_tif_16bit_540p", name[0])
        a_expert = os.path.join(data_path, "gt_16bit_540p", name[1])
        if not os.path.exists(a_input) or not os.path.exists(a_expert):
            continue

        a_input_img = Image.open(a_input)
        a_expert_img = Image.open(a_expert)

        W, H = a_input_img.size
        W2, H2 = a_expert_img.size

        if abs(W - W2) < 2 and abs(H - H2) < 2:
            continue

        skip_names.append(name[0])
        skip_names.append(name[1])

    print('total select {}'.format(len(skip_names)))
    if len(skip_names) > 0:
        with open(os.path.join(data_path, 'skip.txt'), mode='a+') as w:
            for skip_name in set(skip_names):
                w.write('{}\n'.format(skip_name))


def match(data_path):
    singal_match(data_path, 'train_input.txt')
    singal_match(data_path, 'train_label.txt')
    singal_match(data_path, 'test.txt')


def process_label():
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