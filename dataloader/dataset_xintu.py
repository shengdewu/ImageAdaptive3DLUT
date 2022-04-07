import os
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import dataloader.torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF
from dataloader.build import DATASET_ARCH_REGISTRY
from PIL import Image
from dataloader.augmentation import ColorJitter
import logging


def search_files(root, txt, skip_name):
    file = open(os.path.join(root, txt), 'r')
    sorted_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()
    file_names = list()
    for name in sorted_input_files:
        name = name.split(',')
        assert len(name) == 2
        if name[0] in skip_name or name[1] in skip_name:
            continue

        b_input = os.path.join(root, name[0])
        b_expert = os.path.join(root, name[1])
        if not os.path.exists(b_input) or not os.path.exists(b_expert):
            continue
        file_names.append((b_input, b_expert))

    return file_names


@DATASET_ARCH_REGISTRY.register()
class ImageDataSetXinTu(Dataset):
    def __init__(self, cfg, mode="train"):

        root = cfg.DATALOADER.DATA_PATH
        self.mode = mode

        self.skip_name = list()
        if os.path.exists(os.path.join(root, 'skip.txt')):
            file = open(os.path.join(root, 'skip.txt'), 'r')
            self.skip_name = [name.strip('\n') for name in file.readlines()]
            file.close()

        set1_input_files = search_files(root, cfg.DATALOADER.XT_TRAIN_INPUT_TXT, self.skip_name)
        set2_input_files = search_files(root, cfg.DATALOADER.XT_TRAIN_LABEL_TXT, self.skip_name)
        self.test_input_files = search_files(root, cfg.DATALOADER.XT_TEST_TXT, self.skip_name)

        self.set_input_files = set1_input_files + set2_input_files

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]

        self.color_jitter = ColorJitter(cfg.INPUT.COLOR_JITTER, cfg.OUTPUT_LOG_NAME)
        self.color_jitter_prob = cfg.INPUT.COLOR_JITTER.PROB

        self.color_jitter_train = False
        self.train_brightness = 1
        self.train_contrast = 1
        self.train_saturation = 1
        if cfg.INPUT.get('TRAINING_COLOR_JITTER', None) is not None:
            self.color_jitter_train = cfg.INPUT.TRAINING_COLOR_JITTER.ENABLE
            self.train_brightness = cfg.INPUT.TRAINING_COLOR_JITTER.BRIGHTNESS
            self.train_contrast = cfg.INPUT.TRAINING_COLOR_JITTER.CONTRAST
            self.train_saturation = cfg.INPUT.TRAINING_COLOR_JITTER.SATURATION

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('enable {}, training jitter:{}-{}/{}/{}'.format(self.__class__, self.color_jitter_train, self.train_brightness, self.train_contrast, self.train_saturation))
        return

    def __getitem__(self, index):

        if self.mode == 'train':
            input_file, gt_file = self.set_input_files[index % len(self.set_input_files)]
            img_name = os.path.split(input_file)[-1]
        else:
            input_file, gt_file = self.test_input_files[index % len(self.test_input_files)]
            img_name = os.path.split(input_file)[-1]

        img_input = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)
        img_expert = cv2.cvtColor(cv2.imread(gt_file, -1), cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            w = min(img_input.shape[1], img_expert.shape[1])
            h = min(img_input.shape[0], img_expert.shape[0])

            if img_input.shape[:2] != (h, w):
                img_input = img_input[0:h, 0:w, :]
            elif img_expert.shape[:2] != (h, w):
                img_expert = img_expert[0:h, 0:w, :]

            # if img_input.shape[:2] != img_expert.shape[:2]:
            #     ratio_h = np.random.uniform(0.8, 1.0)
            #     ratio_w = np.random.uniform(0.8, 1.0)
            #     w, h = img_input.shape[1], img_input.shape[0]
            #     crop_h = round(h * ratio_h)
            #     crop_w = round(w * ratio_w)
            #     i, j, h, w = TF_x.get_crop_params(img_input, output_size=(crop_h, crop_w))
            #     img_input = TF_x.crop(img_input, i, j, h, w)
            #     img_expert = TF_x.crop(img_expert, i, j, h, w)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF_x.to_tensor(img_expert)

        if self.mode == 'train':
            if np.random.random() <= self.color_jitter_prob:
                img_input = self.color_jitter(img_input)

            if self.color_jitter_train:
                img_expert = TF.adjust_brightness(img_expert, self.train_brightness)
                img_expert = TF.adjust_contrast(img_expert, self.train_contrast)
                img_expert = TF.adjust_saturation(img_expert, self.train_saturation)

        # img_sample = torch.cat((img_input, img_expert_ori, img_expert), -1)
        # ndarr = img_sample.mul(255.0).add_(0.5).clamp_(0, 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
        # cv2.imwrite('/mnt/sda1/enhance.test/image.lut.test/{}.jpg'.format(img_name), ndarr[:, :, ::-1])

        return {"A_input": img_input, "A_exptC": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set_input_files)
        else:
            return len(self.test_input_files)

    def get_item(self, index, skin_name, special_name=None, img_format='jpg'):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file[0])[-1]

        if img_name.endswith('tif') and img_format != 'tif':
            img_name = '{}.{}'.format(img_name[:img_name.rfind('.tif')], img_format)

        if special_name is not None and img_name not in special_name:
            return {"A_input": None, "input_name": img_name}

        if img_name in skin_name:
            return {"A_input": None, "input_name": img_name}

        img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
        img_exptC = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)

        w = min(img_input.shape[1], img_exptC.shape[1])
        h = min(img_input.shape[0], img_exptC.shape[0])

        if img_input.shape[:2] != (h, w):
            img_input = img_input[0:h, 0:w, :]
        elif img_exptC.shape[:2] != (h, w):
            img_exptC = img_exptC[0:h, 0:w, :]

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF_x.to_tensor(img_exptC)

        # down_sample_factor = 4
        # w, h, c = img_input.shape
        # w = (w // down_sample_factor * down_sample_factor)
        # h = (h // down_sample_factor * down_sample_factor)
        # img_input = img_input[:w, :h, :]
        #
        # #intensity = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
        # # img_input = hdr(img_input.astype(np.float32), down_scaler=down_scaler, unnormalizing_value=unnormalizing_value) / unnormalizing_value
        # final_img = run_bf_tone_map(img_input, is_rgb=True, gamma=0.6, down_sample_factor=down_sample_factor)
        # img_input = (final_img * 255.0).astype(np.uint8)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}


@DATASET_ARCH_REGISTRY.register()
class ImageDataSetXinTuUnpaired(Dataset):
    def __init__(self, cfg, mode="train"):

        root = cfg.DATALOADER.DATA_PATH
        self.mode = mode

        self.skip_name = list()
        if os.path.exists(os.path.join(root, 'skip.txt')):
            file = open(os.path.join(root, 'skip.txt'), 'r')
            self.skip_name = [name.strip('\n') for name in file.readlines()]
            file.close()

        self.set1_input_files = search_files(root, cfg.DATALOADER.XT_TRAIN_INPUT_TXT, self.skip_name)
        self.set2_input_files = search_files(root, cfg.DATALOADER.XT_TRAIN_LABEL_TXT, self.skip_name)
        self.test_input_files = search_files(root, cfg.DATALOADER.XT_TEST_TXT, self.skip_name)

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]

        self.color_jitter = ColorJitter(cfg.INPUT.COLOR_JITTER, cfg.OUTPUT_LOG_NAME)
        self.color_jitter_prob = cfg.INPUT.COLOR_JITTER.PROB
        return

    def __getitem__(self, index):

        if self.mode == "train":
            input_file = self.set1_input_files[index % len(self.set1_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)
            seed = random.randint(1, len(self.set2_input_files))
            img2 = cv2.cvtColor(cv2.imread(self.set2_input_files[(index + seed) % len(self.set2_input_files)][1]), cv2.COLOR_BGR2RGB)

        else:
            input_file = self.test_input_files[index % len(self.test_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_exptC = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)
            img2 = img_exptC

        if self.mode == "train":
            if img_exptC.shape[:2] != img_input.shape[:2] or img2.shape[:2] != img_input.shape[:2]:
                ratio_H = np.random.uniform(0.8, 1.0)
                ratio_W = np.random.uniform(0.8, 1.0)
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

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF_x.to_tensor(img_exptC)
        img2 = TF_x.to_tensor(img2)

        if self.mode == 'train':
            if np.random.random() <= self.color_jitter_prob:
                img_input = self.color_jitter(img_input)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        else:
            return len(self.test_input_files)


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


def create_train_label(root_path):
    def label(root_path, paire_name, flag=''):
        index = [i for i in range(len(paire_name))]
        test_name_index = random.sample(index, int(0.2 * len(paire_name)))

        assert len(test_name_index) == len(set(test_name_index)), 'the test name index duplication'

        test_name = list()
        train_name = list()
        for i in index:
            if i in test_name_index:
                test_name.append(paire_name[i])
            else:
                train_name.append(paire_name[i])

        with open(os.path.join(root_path, '{}.test.txt'.format(flag)), mode='w') as t:
            t.write('input,gt\n')
            for name in test_name:
                assert len(name) == 2
                t.write('{},{}\n'.format(name[1], name[0]))

        index = [i for i in range(len(train_name))]
        select_index = random.sample(index, int(0.5 * len(train_name)))

        with open(os.path.join(root_path, '{}.train_input.txt'.format(flag)), mode='w') as ti:
            with open(os.path.join(root_path, '{}.train_label.txt'.format(flag)), mode='w') as tl:
                ti.write('input,gt\n')
                tl.write('input,gt\n')
                for i in index:
                    name = train_name[i]
                    assert len(name) == 2
                    if i in select_index:
                        ti.write('{},{}\n'.format(name[1], name[0]))
                    else:
                        tl.write('{},{}\n'.format(name[1], name[0]))

    with open(os.path.join(root_path, 'gt_16bit_540p.txt'), mode='r') as rgt:
        gts = [r.strip('\n') for r in rgt.readlines()]
    with open(os.path.join(root_path, 'rt_tif_16bit_540p.txt'), mode='r') as igt:
        no_aug_inputs = [r.strip('\n') for r in igt.readlines()]
    with open(os.path.join(root_path, 'rt_tif_aug_16bit_540p.txt'), mode='r') as igt:
        aug_inputs = [r.strip('\n') for r in igt.readlines()]

    aug_inputs = dict([(name.split('/')[1], name) for name in aug_inputs])
    no_aug_inputs = dict([(name.split('/')[1], name) for name in no_aug_inputs])

    aug_inputs.update(no_aug_inputs)

    select_name = list()
    for gt in gts:
        name = gt.split('/')[1]
        zero = '{}_0{}'.format(name[:-4], name[-4:])
        one = '{}_1{}'.format(name[:-4], name[-4:])
        two = '{}_2{}'.format(name[:-4], name[-4:])
        if name not in aug_inputs.keys() or zero not in aug_inputs.keys() or one not in aug_inputs.keys() or two not in aug_inputs.keys():
            print(gt)
            continue
        select_name.append((gt, aug_inputs[name]))
        select_name.append((gt, aug_inputs[zero]))
        select_name.append((gt, aug_inputs[one]))
        select_name.append((gt, aug_inputs[two]))

    label(root_path, select_name, 'all')

    select_name = list()
    for gt in gts:
        name = gt.split('/')[1]
        if name not in no_aug_inputs.keys():
            print(gt)
            continue
        select_name.append((gt, no_aug_inputs[name]))

    label(root_path, select_name, 'no_aug')
