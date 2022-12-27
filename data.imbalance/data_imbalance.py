import os
import numpy as np
import random
import cv2
import tqdm
import json
import configparser
import torch
import torchvision.transforms.functional as ttf
import shutil


"""
resolve missing overexposure data
"""


def bgr2luma(bgr_img: np.ndarray):
    """
    Y = 0.299 R + 0.587 G + 0.114 B
    """
    return 0.299 * bgr_img[:, :, 2] + 0.587 * bgr_img[:, :, 1] + 0.114 * bgr_img[:, :, 0]


def adjust_brightness_adaptive(img: np.ndarray, min_brightness_factor: float, max_brightness_factor: float, max_brightness_stretch: float = 1.0):
    """
    when brightening, the darker is brighter, when press dark, the brighter is darker
    """
    gray = bgr2luma(img) / 65535

    enhance_img = np.zeros_like(img)

    k = (min_brightness_factor - max_brightness_factor) / (max_brightness_stretch-0.0)
    b = min_brightness_factor - max_brightness_stretch * k
    exposure = gray * k + b

    enhance_img[..., 0] = exposure * img[..., 0] + (1.0 - exposure) * enhance_img[..., 0]
    enhance_img[..., 1] = exposure * img[..., 1] + (1.0 - exposure) * enhance_img[..., 1]
    enhance_img[..., 2] = exposure * img[..., 2] + (1.0 - exposure) * enhance_img[..., 2]
    return enhance_img


def adjust_brightness(enhance_img, exposure:float):
    return exposure * enhance_img + (1.0 - exposure) * np.zeros(shape=enhance_img.shape, dtype=enhance_img.dtype)


# 重写optionxform，避免全部改成lower
class NewConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


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
    random.shuffle(file_names)
    return file_names


class GrayDataSet:
    def __init__(self, require_txt:list, data_root_path='/home/shengdewu/data/xt.image.enhancement.540'):

        self.file_list = list()
        for txt_name in require_txt:
            self.file_list.extend(search_files(data_root_path, txt_name, list()))

        self.gray = dict()
        self.over_expose = list()
        self.data_root_path = data_root_path
        return

    def __save__(self, out_path):
        for input_gray, diff_grays in self.gray.items():
            for diff_gray, file_list in diff_grays.items():
                sub_output = os.path.join(out_path, '{}/{}'.format(input_gray, diff_gray))
                os.makedirs(sub_output, exist_ok=True)
                with open('{}/input_gray_{}_diff_gray{}.txt'.format(sub_output, input_gray, diff_gray), mode='w') as w:
                    for i in file_list:
                        w.write('i{},d{},{},{}\n'.format(input_gray, diff_gray, i[0], i[1]))

        for input_gray, diff_grays in self.gray.items():
            with open('{}/{}/diff_gray.txt'.format(out_path, input_gray), mode='w') as w:
                for diff_gray, file_list in diff_grays.items():
                    for i in file_list:
                        w.write('i{},d{},{},{}\n'.format(input_gray, diff_gray, i[0], i[1]))

        diff_gray_list = dict()
        for input_gray, diff_grays in self.gray.items():
            for diff_gray, file_list in diff_grays.items():
                if diff_gray_list.get(diff_gray, None) is None:
                    diff_gray_list[diff_gray] = dict()
                if diff_gray_list[diff_gray].get(input_gray, None) is None:
                    diff_gray_list[diff_gray][input_gray] = list()
                diff_gray_list[diff_gray][input_gray].extend(file_list)

        sgroup = sorted(diff_gray_list.items(), key=lambda kv: kv[0])
        with open('{}/statistics_num.txt'.format(out_path), mode='w') as w:
            for k, v in sgroup:
                cnt = [len(vv) for i, vv in v.items()]
                w.write('{},{}\n'.format(k, sum(cnt)))

        with open('{}/statistics.txt'.format(out_path), mode='w') as w:
            for skv, svv in sgroup:
                for sk, sv in svv.items():
                    for i in sv:
                        w.write('d{}: i{},{},{}\n'.format(skv, sk, i[0], i[1]))

        for input_gray, diff_grays in self.gray.items():
            for diff_gray, file_list in diff_grays.items():
                sub_output = os.path.join(out_path, '{}/{}'.format(input_gray, diff_gray))
                os.makedirs(sub_output, exist_ok=True)
                print(sub_output)

                max_num = min(20, len(file_list))
                for name in tqdm.tqdm(random.sample(file_list, max_num)):
                    img_input = cv2.imread(name[0], cv2.IMREAD_COLOR)
                    img_expert = cv2.imread(name[1], cv2.IMREAD_COLOR)
                    ih, iw, ic = img_input.shape
                    eh, ew, ec = img_expert.shape
                    img_concat = np.zeros(shape=(max(ih, eh), iw+ew, ic), dtype=img_input.dtype)
                    img_concat[:ih, :iw, :] = img_input
                    img_concat[:eh, iw:, :] = img_expert
                    sname, ext = os.path.splitext(os.path.split(name[0])[-1])
                    cv2.imwrite(os.path.join(sub_output, '{}.jpg'.format(sname)), img_concat)

        return

    def __call__(self):
        scale = 255 / 65535 / 10
        scale = 255 / 65535
        index = 0
        for input_file, gt_file in tqdm.tqdm(self.file_list):
            img_input = cv2.imread(input_file, -1)
            img_expert = cv2.imread(gt_file, -1)

            expert_gray = int(np.mean(bgr2luma(img_expert)) * scale + 0.5)
            input_gray = int(np.mean(bgr2luma(img_input)) * scale + 0.5)

            input_name = input_file[input_file.find(self.data_root_path) + len(self.data_root_path) + 1:]
            gt_name = gt_file[gt_file.find(self.data_root_path) + len(self.data_root_path) + 1:]

            if self.gray.get(input_gray, None) is None:
                self.gray[input_gray] = dict()
            if self.gray[input_gray].get(expert_gray, None) is None:
                self.gray[input_gray][expert_gray] = list()

            self.gray[input_gray][expert_gray].append((input_name, gt_name))

            if input_gray > expert_gray:
                self.over_expose.append((input_name, gt_name, input_gray-expert_gray))

            # index += 1
            # if index >= 50:
            #     break

        out_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics'
        os.makedirs(out_path, exist_ok=True)

        over_expose = sorted(self.over_expose, key=lambda item: item[-1])
        with open(os.path.join(out_path, 'no_aug_step1.over_expose.txt'), mode='w') as w:
            w.write('input,gt,diff_gray\n')
            for input_name, gt_name, diff_gray in over_expose:
                w.write('{},{},{}\n'.format(input_name, gt_name, diff_gray))

        gray = sorted(self.gray.items(), key=lambda item: item[0])
        with open(os.path.join(out_path, 'no_aug_step1.gray.txt'), mode='w') as w:
            for i_gray, expert in gray:
                w.write('input gray :{} (*10):\n'.format(i_gray))
                for k, v in expert.items():
                    w.write('   expert:{}; count:{}; files:{}\n'.format(k, len(v), json.dumps(v)))

        # self.__save__(out_path)
        return


def show_over_expose_file():
    txt_name = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/all.over_expose.txt'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/all.over_expose'
    os.makedirs(out_path, exist_ok=True)

    handle = open(txt_name, mode='r')
    names = [name.strip('\n').split(',') for name in handle.readlines()][1:]
    handle.close()

    choices_name = random.choices(names, k=int(0.01*len(names)))
    for in_name, gt_name, value in tqdm.tqdm(choices_name):
        out_name = os.path.join(out_path, '{}-{}'.format(value, in_name.split('/')[-1]))
        if os.path.exists(out_name):
            continue

        in_path = os.path.join(root_path, in_name)
        gt_path = os.path.join(root_path, gt_name)
        in_img = cv2.imread(in_path, -1)
        gt_img = cv2.imread(gt_path, -1)
        h1, w1, c = in_img.shape
        h2, w2, c = gt_img.shape
        concat_img = np.zeros(shape=(max(h1, h2), w1 + w2, c), dtype=in_img.dtype)
        concat_img[:h1, :w1, :] = in_img
        concat_img[:h2, w1:w1+w2, :] = gt_img
        cv2.imwrite(out_name, concat_img)
    return


def show_aug_img():
    rt_name = 'rt_tif_16bit_540p'
    gt_name = 'gt_16bit_540p'
    aug_name = 'rt_tif_aug_16bit_540p'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/aug'
    os.makedirs(out_path, exist_ok=True)

    names = os.listdir(os.path.join(root_path, rt_name))
    choices_name = random.choices(names, k=int(0.005*len(names)))
    for in_name in tqdm.tqdm(choices_name):
        out_name = os.path.join(out_path, in_name)
        if os.path.exists(out_name):
            continue

        in_img = cv2.imread(os.path.join(root_path, rt_name, in_name), -1)
        gt_img = cv2.imread(os.path.join(root_path, gt_name, in_name), -1)
        name = in_name.split('.tif')[0]
        aug0_img = cv2.imread(os.path.join(root_path, aug_name, '{}_0.tif'.format(name)), -1)
        aug1_img = cv2.imread(os.path.join(root_path, aug_name, '{}_1.tif'.format(name)), -1)
        aug2_img = cv2.imread(os.path.join(root_path, aug_name, '{}_2.tif'.format(name)), -1)

        h1, w1, c = in_img.shape
        h2, w2, c = gt_img.shape
        concat_img = np.zeros(shape=(max(h1,h2), w1 * 4+w2, c), dtype=in_img.dtype)
        concat_img[:h1, :w1, :] = in_img
        concat_img[:h1, w1:w1*2, :] = aug0_img
        concat_img[:h1, w1*2:w1*3, :] = aug1_img
        concat_img[:h1, w1*3:w1*4, :] = aug2_img
        concat_img[:h2, w1 * 4:w1 * 4+w2, :] = gt_img
        cv2.imwrite(out_name, concat_img)
    return


def show_aug_img1():
    rt_name = 'rt_tif_16bit_540p'
    gt_name = 'gt_16bit_540p'
    aug_name = 'rt_tif_16bit_imbalance_540p'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/aug2'
    os.makedirs(out_path, exist_ok=True)

    names = os.listdir(os.path.join(root_path, aug_name))
    choices_name = random.choices(names, k=int(0.1*len(names)))
    for in_name in tqdm.tqdm(choices_name):
        name = '{}.tif'.format(in_name.split('-')[0])
        out_name = os.path.join(out_path, name)
        if os.path.exists(out_name):
            continue

        in_img = cv2.imread(os.path.join(root_path, rt_name, name), -1)
        gt_img = cv2.imread(os.path.join(root_path, gt_name, name), -1)
        aug_img = cv2.imread(os.path.join(root_path, aug_name, in_name), -1)

        h1, w1, c = in_img.shape
        h2, w2, c = gt_img.shape
        concat_img = np.zeros(shape=(max(h1, h2), w1 * 2+w2, c), dtype=in_img.dtype)
        concat_img[:h1, :w1, :] = in_img
        concat_img[:h1, w1:w1*2, :] = aug_img
        concat_img[:h2, w1*2:w1*2+w2, :] = gt_img
        cv2.imwrite(out_name, concat_img)
    return


def select_equal_brightness():
    gray_name = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug_step1.gray.txt'
    split_key = 'input gray'
    parse_key = 'expert'
    discard_key = 'rt_tif_aug_16bit_540p'
    i_gray = None
    select_names = list()
    equal_names = list()
    over_names = list()
    under_names = list()
    with open(gray_name, mode='r') as handle:
        while True:
            line = handle.readline()
            if line is None or len(line) == 0:
                break
            line = line.strip('\n').strip(' ')
            if line.find(split_key) != -1:
                print('start process {}'.format(line))
                arr = line.split(':')
                i_gray = int(arr[1].split(' ')[0].strip(' '))
                continue

            assert line.find(parse_key) != -1 and i_gray is not None, 'the {} is invalid parse'.format(line)
            arr = line.split(';')
            o_gray = int(arr[0].split(':')[-1].strip(' '))
            diff = i_gray - o_gray

            json_file = arr[-1].split(':')[-1]
            for rt, gt in json.loads(json_file):
                if rt.find(discard_key) != -1:
                    continue

                if abs(diff) <= 5:
                    equal_names.append((rt, gt, diff, i_gray))
                elif diff > 0:
                    over_names.append((rt, gt, diff, i_gray))
                else:
                    under_names.append((rt, gt, diff, i_gray))

                if -10 < diff <= -1:
                    select_names.append((rt, gt, diff, i_gray))

    # with open('/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.select.txt', mode='w') as handle:
    #     handle.write('input,gt,diff_gray,i_gray\n')
    #     for rt, gt, diff, i_gray in select_names:
    #         handle.write('{},{},{},{}\n'.format(rt, gt, diff, i_gray))

    with open('no_aug.equal_bright.txt', mode='w') as handle:
        handle.write('input,gt,diff_gray,i_gray\n')
        for rt, gt, diff, i_gray in equal_names:
            handle.write('{},{},{},{}\n'.format(rt, gt, diff, i_gray))

    with open('no_aug.under_bright.txt', mode='w') as handle:
        handle.write('input,gt,diff_gray,i_gray\n')
        for rt, gt, diff, i_gray in under_names:
            handle.write('{},{},{},{}\n'.format(rt, gt, diff, i_gray))

    with open('no_aug.over_bright.txt', mode='w') as handle:
        handle.write('input,gt,diff_gray,i_gray\n')
        for rt, gt, diff, i_gray in over_names:
            handle.write('{},{},{},{}\n'.format(rt, gt, diff, i_gray))

    return


def add_over_expose_by_rt():
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_sub_path = 'rt_tif_16bit_imbalance_540p'
    out_root = os.path.join(root_path, out_sub_path)
    os.makedirs(out_root, exist_ok=True)

    select_name_txt = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.select.txt'

    rt_cli = '/home/shengdewu/workspace/RawTherapee/build/rtgui/rawtherapee-cli'
    default_pp3 = './rt/over.pp3'

    handle = open(select_name_txt, mode='r')
    lines = [name.strip('\n') for name in handle.readlines()]
    handle.close()
    assert lines[0] == 'input,gt,diff_gray,i_gray'

    for line in lines[1:]:
        in_name, gt_name, diff, i_gray = line.split(',')
        in_path = os.path.join(root_path, in_name)
        gt_path = os.path.join(root_path, gt_name)
        assert os.path.exists(in_path) and os.path.exists(gt_path)
        name = in_name.split('/')[-1]
        assert name == gt_name.split('/')[-1]

        exposure = random.randint(1, 5)

        arr = name.split('.')
        out_path = os.path.join(out_root, '{}_{}.{}'.format(arr[0], exposure, arr[1]))

        cf = NewConfigParser()
        cf.read(default_pp3)
        # cf.set('White Balance', 'Setting', 'Custom')

        pp3 = os.path.join(out_root, '{}_{}.{}.pp3'.format(arr[0], exposure, arr[1]))

        exposure = exposure * 0.15
        cf.set('Exposure', 'Compensation', str(exposure))
        with open(pp3, 'w') as f:
            cf.write(f, space_around_delimiters=False)

        cmd = rt_cli + ' -o ' + out_path + ' -b16 -t -q -f -p ' + pp3 + ' -c ' + in_path
        os.system(cmd)
        os.remove(pp3)
    return


def add_over_expose():

    max_value = 65535
    scale = 255 / 65535

    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_sub_path = 'rt_tif_16bit_imbalance_540p'
    out_root = os.path.join(root_path, out_sub_path)
    os.makedirs(out_root, exist_ok=True)

    select_name_txt = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.select.txt'

    skip_name = 'skip.txt'
    h = open(os.path.join(root_path, skip_name), mode='r')
    skip_names = [name.strip('\n') for name in h.readlines()]
    h.close()

    handle = open(select_name_txt, mode='r')
    lines = [name.strip('\n') for name in handle.readlines()]
    handle.close()
    assert lines[0] == 'input,gt,diff_gray,i_gray'
    lines = lines[1:]
    random.shuffle(lines)
    for line in tqdm.tqdm(lines):
        in_name, gt_name, diff, i_gray = line.split(',')
        if in_name in skip_names or gt_name in skip_names:
            continue
        i_gray = int(i_gray)
        diff = int(diff)

        if i_gray > 200 or i_gray < 20:
            continue

        g_gray = i_gray - diff

        if g_gray / i_gray >= 1.5:
            continue

        in_path = os.path.join(root_path, in_name)
        gt_path = os.path.join(root_path, gt_name)
        assert os.path.exists(in_path) and os.path.exists(gt_path)
        name = in_name.split('/')[-1]
        assert name == gt_name.split('/')[-1]

        in_img = cv2.imread(in_path, -1)
        float_img = in_img.astype(np.float32)
        arr = name.split('.')
        exposure = 1 + (abs(diff) + 0.1) * 0.01
        s = 1
        enhance_img = float_img.copy()
        while True:
            enhance_img = np.clip(adjust_brightness(enhance_img, exposure), 0, max_value).astype(in_img.dtype)
            # enhance_img = np.clip(adjust_brightness_adaptive(enhance_img, 1.0, exposure), 0, max_value).astype(in_img.dtype)
            gray = round(np.mean(bgr2luma(enhance_img)) * scale, 2)
            if gray - g_gray > 5 or exposure > 1.3:
                break
            exposure += 0.001 * s
            s /= 2
            # exposure += 0.001

        # out_path = os.path.join(out_root, '{}-{}_{}_{}_{}.{}'.format(arr[0], int(gray), round(exposure, 3), i_gray, g_gray, arr[1]))
        #
        # gt_img = cv2.imread(gt_path, -1)
        #
        # h1, w1, c = in_img.shape
        # h2, w2, c = gt_img.shape
        # concat_img = np.zeros(shape=(max(h1, h2), w1 * 2+w2, c), dtype=in_img.dtype)
        # concat_img[:h1, :w1, :] = in_img
        # concat_img[:h1, w1:w1*2, :] = enhance_img
        # concat_img[:h2, w1*2:w1*2+w2, :] = gt_img
        # cv2.imwrite(out_path, concat_img)

        out_path = os.path.join(out_root, '{}-{}.{}'.format(arr[0], str(round((gray-g_gray), 1)).replace('.', '_'), arr[1]))
        cv2.imwrite(out_path, enhance_img)

    return


def create_label():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        assert names[0] == 'input,gt,diff_gray,i_gray'
        return names[1:]

    imbalance_sub = 'rt_tif_16bit_imbalance_540p'
    gt_sub = 'gt_16bit_540p'
    rt_sub = 'rt_tif_16bit_540p'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    over_name_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.over_bright.txt'
    equal_name_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.equal_bright.txt'
    under_name_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.under_bright.txt'

    imbalance_name = list()
    for im_name in os.listdir(os.path.join(root_path, imbalance_sub)):
        arr = im_name.split('-')
        name = '{}.{}'.format(arr[0], arr[1].split('.')[-1])
        gt_name = os.path.join(gt_sub, name)
        rt_name = os.path.join(rt_sub, name)
        assert os.path.exists(os.path.join(root_path, gt_name)) and os.path.exists(os.path.join(root_path, rt_name))
        imbalance_name.append((os.path.join(imbalance_sub, im_name), gt_name))

    with open(os.path.join(root_path, 'imbalance_gt.txt'), mode='w') as handle:
        handle.write('input,gt\n')
        for im_name, gt_name in imbalance_name:
            handle.write('{},{}\n'.format(im_name, gt_name))

    over_name = get_names(over_name_path)
    equal_name = get_names(equal_name_path)
    over_name.extend(equal_name)
    under_names = get_names(under_name_path)
    # select_under_name = int((len(over_name) + len(imbalance_name)) * 2 + 0.5)
    # under_name = random.sample(under_names, k=min(len(under_names), select_under_name))
    over_name.extend(under_names)

    skip_name = 'skip.txt'
    h = open(os.path.join(root_path, skip_name), mode='r')
    skip_names = [name.strip('\n') for name in h.readlines()]
    h.close()

    all_names = list()
    for name in over_name:
        in_name, gt_name, diff, i_gray = name.split(',')
        if in_name in skip_names or gt_name in skip_names:
            continue
        assert os.path.exists(os.path.join(root_path, in_name)) and os.path.exists(os.path.join(root_path, gt_name))
        all_names.append((in_name, gt_name))

    all_names.extend(imbalance_name)
    random.shuffle(all_names)
    index = [i for i in range(len(all_names))]
    train_index = random.sample(index, k=int(0.9*len(index)))

    train_names = list()
    test_names = list()
    for i in index:
        if i in train_index:
            train_names.append(all_names[i])
        else:
            test_names.append(all_names[i])

    random.shuffle(train_names)

    with open(os.path.join(root_path, 'im.train_input.txt'), mode='w') as hd:
        hd.write('input,gt\n')
        for name in train_names[:int(len(train_names) * 0.5)]:
            hd.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(root_path, 'im.train_label.txt'), mode='w') as hd:
        hd.write('input,gt\n')
        for name in train_names[int(len(train_names) * 0.5):]:
            hd.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(root_path, 'im.test.txt'), mode='w') as hd:
        hd.write('input,gt\n')
        for name in test_names:
            hd.write('{},{}\n'.format(name[0], name[1]))
    return


def cp():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        assert names[0] == 'input,gt'
        return names[1:]

    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    train_name = 'im.train_label.txt'
    input_name = 'im.train_input.txt'
    test_name = 'im.test.txt'

    with open(os.path.join(root_path, 'imbalance_gt.txt'), mode='w') as handle:
        handle.write('input,gt\n')
        for im_name, gt_name in imbalance_name:
            handle.write('{},{}\n'.format(im_name, gt_name))


    names = get_names(os.path.join(root_path, train_name))
    names.extend(get_names(os.path.join(root_path, input_name)))
    names.extend(get_names(os.path.join(root_path, test_name)))
    out_path = '/home/shengdewu/data/xt.image.enhancement.540'
    for name in tqdm.tqdm(names):
        rt_name, gt_name = name.split(',')

        if os.path.exists(os.path.join(out_path, rt_name)) and os.path.exists(os.path.join(out_path, gt_name)):
            continue

        shutil.copy(os.path.join(root_path, rt_name), os.path.join(out_path, rt_name))
        shutil.copy(os.path.join(root_path, gt_name), os.path.join(out_path, gt_name))

    shutil.copy(os.path.join(root_path, train_name), os.path.join(out_path, train_name))
    shutil.copy(os.path.join(root_path, input_name), os.path.join(out_path, input_name))
    shutil.copy(os.path.join(root_path, test_name), os.path.join(out_path, test_name))
    return


def check():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        assert names[0] == 'input,gt'
        return names[1:]

    root_path = '/home/shengdewu/data/xt.image.enhancement.540'
    train_name = 'im.train_label.txt'
    input_name = 'im.train_input.txt'
    test_name = 'im.test.txt'

    names = get_names(os.path.join(root_path, train_name))
    names.extend(get_names(os.path.join(root_path, input_name)))
    names.extend(get_names(os.path.join(root_path, test_name)))
    for name in tqdm.tqdm(names):
        rt_name, gt_name = name.split(',')
        try:
            img_input = cv2.cvtColor(cv2.imread(os.path.join(root_path, rt_name), -1), cv2.COLOR_BGR2RGB)
            img_expert = cv2.cvtColor(cv2.imread(os.path.join(root_path, gt_name), -1), cv2.COLOR_BGR2RGB)
        except Exception as err:
            print('{}-{}'.format(rt_name, gt_name))

    return


def show_im_img():
    root_path = '/home/shengdewu/data/xt.image.enhancement.540.jpg'
    im_names = ['im.train_input.txt', 'im.train_label.txt', 'im.test.txt']
    out_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/aug2'
    os.makedirs(out_path, exist_ok=True)
    names = list()
    for im_name in im_names:
        handle = open(os.path.join(root_path, im_name), mode='r')
        names.extend([name.strip('\n') for name in handle.readlines()][1:])
        handle.close()

    choices_name = random.sample(names, k=int(0.1*len(names)))
    for in_name in tqdm.tqdm(choices_name):
        rt_name, gt_name = in_name.split(',')
        out_name = rt_name.split('/')[-1]
        in_img = cv2.imread(os.path.join(root_path, rt_name), cv2.IMREAD_COLOR)
        gt_img = cv2.imread(os.path.join(root_path, gt_name), cv2.IMREAD_COLOR)

        h1, w1, c = in_img.shape
        h2, w2, c = gt_img.shape
        concat_img = np.zeros(shape=(max(h1, h2), w1+w2, c), dtype=in_img.dtype)
        concat_img[:h1, :w1, :] = in_img
        concat_img[:h2, w1:w1+w2, :] = gt_img
        cv2.imwrite(os.path.join(out_path, out_name.replace('tif', 'jpg')), concat_img)
    return


def bright_rt_img():
    max_value = 65535
    scale = 255 / 65535

    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_sub_path = 'rt_tif_16bit_no_gt_imbalance_540p'
    out_root = os.path.join(root_path, out_sub_path)
    os.makedirs(out_root, exist_ok=True)

    select_name_txt = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/no_aug.under_bright.txt'

    skip_name = 'skip.txt'
    h = open(os.path.join(root_path, skip_name), mode='r')
    skip_names = [name.strip('\n') for name in h.readlines()]
    h.close()

    handle = open(select_name_txt, mode='r')
    lines = [name.strip('\n') for name in handle.readlines()]
    handle.close()
    assert lines[0] == 'input,gt,diff_gray,i_gray'
    lines = lines[1:]
    random.shuffle(lines)

    filter_lines = list()
    for line in tqdm.tqdm(lines):
        in_name, gt_name, diff, i_gray = line.split(',')
        if in_name in skip_names or gt_name in skip_names:
            continue
        i_gray = int(i_gray)
        if i_gray > 200 or i_gray < 20:
            continue
        filter_lines.append(line)

    choices_lines = random.sample(filter_lines, k=int(0.3*len(filter_lines)))
    for line in tqdm.tqdm(choices_lines):
        in_name, gt_name, diff, i_gray = line.split(',')
        if in_name in skip_names or gt_name in skip_names:
            continue
        i_gray = int(i_gray)
        diff = int(diff)

        if i_gray > 200 or i_gray < 20:
            continue

        g_gray = i_gray - diff

        in_path = os.path.join(root_path, in_name)
        gt_path = os.path.join(root_path, gt_name)
        assert os.path.exists(in_path) and os.path.exists(gt_path)
        name = in_name.split('/')[-1]
        assert name == gt_name.split('/')[-1]

        in_img = cv2.imread(in_path, -1)
        float_img = in_img.astype(np.float32)
        arr = name.split('.')
        exposure = 1.01
        s = 1
        enhance_img = float_img.copy()
        while True:
            enhance_img = np.clip(adjust_brightness(enhance_img, exposure), 0, max_value).astype(in_img.dtype)
            # enhance_img = np.clip(adjust_brightness_adaptive(enhance_img, 1.0, exposure), 0, max_value).astype(in_img.dtype)
            gray = round(np.mean(bgr2luma(enhance_img)) * scale, 2)
            if gray - i_gray > 20 or exposure > 1.21:
                break
            exposure += 0.001 * s
            s /= 2
            # exposure += 0.001

        # out_path = os.path.join(out_root, '{}-{}_{}_{}_{}.{}'.format(arr[0], int(gray), round(exposure, 3), i_gray, g_gray, arr[1]))
        #
        # gt_img = cv2.imread(gt_path, -1)
        #
        # h1, w1, c = in_img.shape
        # h2, w2, c = gt_img.shape
        # concat_img = np.zeros(shape=(max(h1, h2), w1 * 2+w2, c), dtype=in_img.dtype)
        # concat_img[:h1, :w1, :] = in_img
        # concat_img[:h1, w1:w1*2, :] = enhance_img
        # concat_img[:h2, w1*2:w1*2+w2, :] = gt_img
        # cv2.imwrite(out_path, concat_img)

        out_path = os.path.join(out_root, '{}-{}.{}'.format(arr[0], str(round((gray-g_gray), 1)).replace('-', 'f').replace('.', '_'), arr[1]))
        cv2.imwrite(out_path, enhance_img)

    return


def class_img():
    out_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    rt_sub = 'rt_tif_16bit_540p'
    gt_sub = 'gt_16bit_540p'

    cls_names = dict()
    for name in tqdm.tqdm(os.listdir(os.path.join(root_path, rt_sub))):
        cls_name = name.split('_')[0]
        if cls_names.get(cls_name, None) is None:
            cls_names[cls_name] = list()
        cls_names[cls_name].append((os.path.join(rt_sub, name), os.path.join(gt_sub, name)))
        # shutil.copy(os.path.join(root_path, rt_sub, name), os.path.join(out_root, cls_name, rt_sub, name))
        # shutil.copy(os.path.join(root_path, gt_sub, name), os.path.join(out_root, cls_name, gt_sub, name))

    os.makedirs(out_root, exist_ok=True)
    for k, names in cls_names.items():
        with open(os.path.join(out_root, '{}.txt'.format(k)), mode='w') as f:
            for name in names:
                f.write('{},{}\n'.format(name[0], name[1]))

    total_nums = sum([len(v) for k, v in cls_names.items()])
    print(total_nums)
    total_key = sorted(cls_names.items(), key=lambda item:len(item[1]))
    with open(os.path.join(out_root, 'total-{}.txt'.format(total_nums)), mode='w') as f:
        for k, v in total_key:
            f.write('{}: {}\n'.format(k, len(v)))
    return


def select_class_img():
    cls_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    out_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.select.540'
    rt_sub = 'rt_tif_16bit_540p'
    gt_sub = 'gt_16bit_540p'

    total_nums = 0
    txt_names = os.listdir(cls_root)
    for txt_name in txt_names:
        if txt_name.find('total-') != -1:
            continue
        f = open(os.path.join(cls_root, txt_name), mode='r')
        names = [name.strip().split(',') for name in f.readlines()]
        f.close()

        cls_out_path = os.path.join(out_root, txt_name.split('.')[0])
        rt_out_path = os.path.join(cls_out_path, rt_sub)
        gt_out_path = os.path.join(cls_out_path, gt_sub)
        # os.makedirs(rt_out_path, exist_ok=True)
        os.makedirs(gt_out_path, exist_ok=True)

        select_names = random.sample(names, k=max(min(10, len(names)), int(len(names) * 0.25)))

        for name in select_names:
            assert name[0].find(rt_sub) != -1 and name[1].find(gt_sub) != -1
            # shutil.copy(os.path.join(root_path, name[0]), os.path.join(cls_out_path, name[0]))
            shutil.copy(os.path.join(root_path, name[1]), os.path.join(cls_out_path, name[1]))

        total_nums += len(os.listdir(gt_out_path))
    
    print(total_nums)
    return


def select_img():
    out_root = '/mnt/sdb/data.set/xintu.data/enhance.data/txt'
    root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    rt_sub = 'rt_tif_16bit_540p'
    gt_sub = 'gt_16bit_540p'
    nums = 5
    
    total_key = dict()

    for sub_dir in tqdm.tqdm(os.listdir(root_path)): 
        if sub_dir.endswith('txt'):
            continue
        sub_gt_path = os.path.join(root_path, sub_dir, gt_sub)
        names = os.listdir(sub_gt_path)
        total_key[sub_dir] = len(names)
        # for name in random.sample(names, k=min(len(names), nums)):
        #     shutil.copy(os.path.join(sub_gt_path, name), os.path.join(out_root, name))

    total_key = sorted(total_key.items(), key=lambda item:item[1])
    with open(os.path.join(root_path, 'cls.txt'), mode='w') as f:
        for k, v in total_key:
            f.write('{}: {}\n'.format(k, v))


def add_over_expose_by_gt():

    max_value = 65535
    scale = 255 / 65535

    def bright(img_path):
        in_img = cv2.imread(img_path, -1)
        float_img = in_img.astype(np.float32)
        
        gray = round(np.mean(bgr2luma(float_img)) * scale, 2)
        max_factor = 22
        if gray >= 200:
            max_factor = 10
        exposure = 1 + random.randrange(5, max_factor, 2) / 100
        # print(exposure)
        enhance_img = np.clip(adjust_brightness(float_img, exposure), 0, max_value).astype(in_img.dtype)
        return enhance_img, exposure

    out_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.over.540'
    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    cls_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    gt_sub = 'gt_16bit_540p'
    rt_sub = 'rt_tif_16bit_540p'

    os.makedirs(out_root_path, exist_ok=True)

    over_name = 'over'
    gt_name = 'gt'
    total_nums = 0
    txt_names = os.listdir(cls_root)
    for txt_name in txt_names:
        if txt_name.find('total-') != -1:
            continue
        f = open(os.path.join(cls_root, txt_name), mode='r')
        names = [name.strip().split(',') for name in f.readlines()]
        f.close()

        # over_out_path = os.path.join(out_root_path, txt_name.split('.')[0], over_name)
        # gt_out_path = os.path.join(out_root_path, txt_name.split('.')[0], gt_name)
        # os.makedirs(gt_out_path, exist_ok=True)
        # os.makedirs(over_out_path, exist_ok=True)

        select_names = random.sample(names, k=max(min(10, len(names)), int(len(names) * 0.2)))
        for name in select_names:
            assert name[0].find(rt_sub) != -1 and name[1].find(gt_sub) != -1
            gname = os.path.join(in_root_path, name[1])
            oname = name[1].split('/')[-1]
            enhance_img, exposure = bright(gname)
            cv2.imwrite(os.path.join(out_root_path, oname), enhance_img)
            # shutil.copy(gname, os.path.join(gt_out_path, oname))

        # total_nums += len(os.listdir(over_out_path))
    print(len(os.listdir(out_root_path)))
    return


def split_over_expose_by_gt():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        return names

    def select_names(name_type='no_aug.under_bright.txt'):
        select_names_pair = list()
        remind_keys = list()
        equal_name_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/{}'.format(name_type)
        with open(equal_name_path, mode='r') as f:
            names = [name.strip('\n') for name in f.readlines()]
            assert names[0] == 'input,gt,diff_gray,i_gray'
            for name in names[1:]:
                arr = name.split(',')
                key = arr[0].split('/')[-1]
                assert key not in remind_keys
                remind_keys.append(key)
                select_names_pair.append((arr[0], arr[1]))
                # print('{}-{}'.format(arr[0], arr[1]))
        return select_names_pair,  remind_keys

    # select_names_pair, remind_keys = select_names()
    #
    # print('select_names_pair = {}'.format(len(select_names_pair)))

    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    gt_txt = 'gt_16bit_540p.txt'
    rt_txt = 'rt_tif_16bit_540p.txt'
    over_txt = 'rt_16bit_over_540p.txt'

    gt_names = get_names(os.path.join(in_root_path, gt_txt))
    rt_names = get_names(os.path.join(in_root_path, rt_txt))
    over_names = get_names(os.path.join(in_root_path, over_txt))

    print('over = {}'.format(len(over_names)))

    over_keys_names = dict([(name.split('/')[-1], name) for name in over_names])
    gt_keys_names = dict([(name.split('/')[-1], name) for name in gt_names])
    rt_keys_names = dict([(name.split('/')[-1], name) for name in rt_names])

    names_pair = list()
    for key, name in over_keys_names.items():
        names_pair.append((name, gt_keys_names[key]))

    print('over pair {}'.format(len(names_pair)))

    # names_pair = random.sample(select_names_pair, k=min(len(over_names_pair), int(len(select_names_pair)*0.5)))
    #
    # names_pair.extend(over_names_pair)

    # remind_keys.extend(over_keys_names.keys())

    index = 0
    rt_names_pair = list()
    for key, name in rt_keys_names.items():
        # if key in over_keys_names.keys():
        #     index += 1
        #     continue
        if not os.path.exists(os.path.join(in_root_path, name)) or not os.path.exists(os.path.join(in_root_path, gt_keys_names[key])):
            continue
        rt_names_pair.append((name, gt_keys_names[key]))
    
    print('rt pair {}, rt {}'.format(len(rt_names_pair), len(rt_keys_names)))
    
    # index = [i for i in range(len(rt_names_pair))]
    # random.shuffle(rt_names_pair)
    rt_names_pair = random.sample(rt_names_pair, k=int(len(names_pair)*2))
    print('select rt {}'.format(len(rt_names_pair)))
    names_pair.extend(rt_names_pair)
    print('total pair {}'.format(len(names_pair)))

    out_root_path = '/home/shengdewu/data/xt.image.enhancement.540'

    random.shuffle(names_pair)
    test_nums = int(0.05 * len(names_pair))
    test_names = names_pair[: test_nums]
    train_names = names_pair[test_nums: ]
    train_point = int(0.5 * len(train_names))
    print('train {}, test {}'.format(len(train_names), len(test_names)))
    with open(os.path.join(out_root_path, 'im.test.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in test_names:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'im.train_input.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[: train_point]:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'im.train_label.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[train_point :]:
            f.write('{},{}\n'.format(name[0], name[1]))

    tmp_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.tmp'
    os.makedirs(os.path.join(tmp_root_path, 'gt_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_tif_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    os.makedirs(os.path.join(out_root_path, 'gt_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_tif_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    for name in tqdm.tqdm(names_pair):
        # print('{} -> {}'.format(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0])))
        # print('{} -> {}'.format(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1])))
        if os.path.exists(os.path.join(out_root_path, name[0])) and os.path.exists(os.path.join(out_root_path, name[1])):
            continue

        shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0]))
        shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1]))
        if random.random() <= 0.01:
            shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(tmp_root_path, name[0]))
            shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(tmp_root_path, name[1]))            
    return


def split_only_over_expose_by_gt():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        return names

    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    gt_txt = 'gt_16bit_540p.txt'
    over_txt = 'rt_16bit_over_540p.txt'

    gt_names = get_names(os.path.join(in_root_path, gt_txt))
    over_names = get_names(os.path.join(in_root_path, over_txt))

    print('over = {}'.format(len(over_names)))

    over_keys_names = dict([(name.split('/')[-1], name) for name in over_names])
    gt_keys_names = dict([(name.split('/')[-1], name) for name in gt_names])

    over_names_pair = list()
    for key, name in over_keys_names.items():
        over_names_pair.append((name, gt_keys_names[key]))

    print('over pair {}'.format(len(over_names_pair)))

    out_root_path = '/home/shengdewu/data/xt.image.enhancement.540'

    random.shuffle(over_names_pair)
    test_nums = int(0.05 * len(over_names_pair))
    test_names = over_names_pair[: test_nums]
    train_names = over_names_pair[test_nums: ]
    train_point = int(0.5 * len(train_names))
    print('train {}, test {}'.format(len(train_names), len(test_names)))
    with open(os.path.join(out_root_path, 'over.test.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in test_names:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'over.train_input.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[: train_point]:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'over.train_label.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[train_point :]:
            f.write('{},{}\n'.format(name[0], name[1]))

    # tmp_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.tmp'
    # os.makedirs(os.path.join(tmp_root_path, 'gt_16bit_540p'), exist_ok=True)
    # os.makedirs(os.path.join(tmp_root_path, 'rt_tif_16bit_540p'), exist_ok=True)
    # os.makedirs(os.path.join(tmp_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    # os.makedirs(os.path.join(out_root_path, 'gt_16bit_540p'), exist_ok=True)
    # os.makedirs(os.path.join(out_root_path, 'rt_tif_16bit_540p'), exist_ok=True)
    # os.makedirs(os.path.join(out_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    # for name in tqdm.tqdm(over_names_pair):
    #     # print('{} -> {}'.format(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0])))
    #     # print('{} -> {}'.format(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1])))
    #     if os.path.exists(os.path.join(out_root_path, name[0])) and os.path.exists(os.path.join(out_root_path, name[1])):
    #         continue
    #     shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0]))
    #     shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1]))
    #     if random.random() <= 0.01:
    #         shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(tmp_root_path, name[0]))
    #         shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(tmp_root_path, name[1]))            
    return


def add_under_expose_by_gt():

    max_value = 65535
    scale = 255 / 65535

    def bright(img_path):
        in_img = cv2.imread(img_path, -1)
        float_img = in_img.astype(np.float32)
        
        gray = round(np.mean(bgr2luma(float_img)) * scale, 2)
        max_factor = 20
        min_factor = 5
        if gray >= 150:
            max_factor = 40
            min_factor = 10
        exposure = 1 - random.randrange(min_factor, max_factor, 5) / 100
        # print(exposure)
        enhance_img = np.clip(adjust_brightness(float_img, exposure), 0, max_value).astype(in_img.dtype)
        return enhance_img, exposure

    out_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540/rt_16bit_under_540p'
    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    cls_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    gt_sub = 'gt_16bit_540p'
    rt_sub = 'rt_tif_16bit_540p'

    os.makedirs(out_root_path, exist_ok=True)

    over_name = 'over'
    gt_name = 'gt'
    total_nums = 0
    txt_names = os.listdir(cls_root)
    random.shuffle(txt_names)
    
    for txt_name in txt_names:
        if txt_name.find('total-') != -1:
            continue
        f = open(os.path.join(cls_root, txt_name), mode='r')
        names = [name.strip().split(',') for name in f.readlines()]
        f.close()

        # over_out_path = os.path.join(out_root_path, txt_name.split('.')[0], over_name)
        # gt_out_path = os.path.join(out_root_path, txt_name.split('.')[0], gt_name)
        # os.makedirs(gt_out_path, exist_ok=True)
        # os.makedirs(over_out_path, exist_ok=True)

        select_names = random.sample(names, k=max(min(10, len(names)), int(len(names) * 0.3)))
        for name in select_names:
            assert name[0].find(rt_sub) != -1 and name[1].find(gt_sub) != -1
            gname = os.path.join(in_root_path, name[1])
            oname = name[1].split('/')[-1]
            enhance_img, exposure = bright(gname)
            cv2.imwrite(os.path.join(out_root_path, oname), enhance_img)
            # shutil.copy(gname, os.path.join(gt_out_path, oname))

        # total_nums += len(os.listdir(over_out_path))
    print(len(os.listdir(out_root_path)))
    return


def split_only_over_under_expose_by_gt():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        return names

    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540'
    gt_txt = 'gt_16bit_540p.txt'
    over_txt = 'rt_16bit_over_540p.txt'
    under_txt = 'rt_16bit_under_540p.txt'
    rt_txt = 'rt_tif_16bit_540p.txt'

    gt_names = get_names(os.path.join(in_root_path, gt_txt))
    over_names = get_names(os.path.join(in_root_path, over_txt))
    under_names = get_names(os.path.join(in_root_path, rt_txt))

    under_names = random.sample(under_names, k=int(len(over_names)*1.2))
    print('over = {}'.format(len(over_names)))
    print('under = {}'.format(len(under_names)))

    over_keys_names = dict([(name.split('/')[-1], name) for name in over_names])
    under_keys_names = dict([(name.split('/')[-1], name) for name in under_names])
    gt_keys_names = dict([(name.split('/')[-1], name) for name in gt_names])

    names_pair = list()
    for key, name in over_keys_names.items():
        names_pair.append((name, gt_keys_names[key]))

    for key, name in under_keys_names.items():
        names_pair.append((name, gt_keys_names[key]))

    print('over pair {}'.format(len(names_pair)))

    out_root_path = '/home/shengdewu/data/xt.image.enhancement.540'

    random.shuffle(names_pair)
    test_nums = int(0.05 * len(names_pair))
    test_names = names_pair[: test_nums]
    train_names = names_pair[test_nums: ]
    train_point = int(0.5 * len(train_names))
    print('train {}, test {}'.format(len(train_names), len(test_names)))
    with open(os.path.join(out_root_path, 'over_under.test.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in test_names:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'over_under.train_input.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[: train_point]:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'over_under.train_label.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[train_point :]:
            f.write('{},{}\n'.format(name[0], name[1]))

    tmp_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.tmp'
    os.makedirs(os.path.join(tmp_root_path, 'gt_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_16bit_under_540p'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    os.makedirs(os.path.join(out_root_path, 'gt_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_16bit_under_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_16bit_over_540p'), exist_ok=True)

    for name in tqdm.tqdm(names_pair):
        # print('{} -> {}'.format(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0])))
        # print('{} -> {}'.format(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1])))
        if os.path.exists(os.path.join(out_root_path, name[0])) and os.path.exists(os.path.join(out_root_path, name[1])):
            continue
        shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0]))
        shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1]))
        if random.random() <= 0.01:
            shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(tmp_root_path, name[0]))
            shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(tmp_root_path, name[1]))            
    return


def add_over_expose_by_gt_jpg():

    max_value = 255

    def bright(img_path):
        in_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        float_img = in_img.astype(np.float32)
        
        gray = round(np.mean(bgr2luma(float_img)), 2)

        max_factor = 22
        if gray >= 200:
            max_factor = 10
        exposure = 1 + random.randrange(5, max_factor, 2) / 100
        # print(exposure)
        enhance_img = np.clip(adjust_brightness(float_img, exposure), 0, max_value).astype(in_img.dtype)
        return enhance_img, exposure

    out_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.jpg/rt_8bit_only_adjust_light_over_1500p'
    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.jpg'
    cls_root = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.cls.540'
    gt_sub = 'gt_16bit_540p'
    rt_sub = 'rt_tif_16bit_540p'

    os.makedirs(out_root_path, exist_ok=True)

    over_name = 'over'
    gt_name = 'gt'
    total_nums = 0
    txt_names = os.listdir(cls_root)
    for txt_name in txt_names:
        if txt_name.find('total-') != -1:
            continue
        f = open(os.path.join(cls_root, txt_name), mode='r')
        names = [name.strip().split(',') for name in f.readlines()]
        f.close()

        # over_out_path = os.path.join(out_root_path, txt_name.split('.')[0], over_name)
        # gt_out_path = os.path.join(out_root_path, txt_name.split('.')[0], gt_name)
        # os.makedirs(gt_out_path, exist_ok=True)
        # os.makedirs(over_out_path, exist_ok=True)

        select_names = random.sample(names, k=max(min(10, len(names)), int(len(names) * 0.2)))
        for name in select_names:
            assert name[0].find(rt_sub) != -1 and name[1].find(gt_sub) != -1
            oname = name[1].split('/')[-1].replace('.tif', '.jpg')
            gname = os.path.join(in_root_path, 'gt_8bit_1500p_only_adjust_light', oname)

            enhance_img, exposure = bright(gname)
            cv2.imwrite(os.path.join(out_root_path, oname), enhance_img)
            # shutil.copy(gname, os.path.join(gt_out_path, oname))

        # total_nums += len(os.listdir(over_out_path))
    print(len(os.listdir(out_root_path)))
    return


def split_over_expose_by_gt_jpg():
    def get_names(path):
        h = open(path, mode='r')
        names = [name.strip('\n') for name in h.readlines()]
        h.close()
        return names

    # def select_names(name_type='no_aug.under_bright.txt'):
    #     root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.jpg'
    #     select_names_pair = list()
    #     remind_keys = list()
    #     equal_name_path = '/mnt/sdb/data.set/xintu.data/enhance.data/img.gray.statistics/{}'.format(name_type)
    #     with open(equal_name_path, mode='r') as f:
    #         names = [name.strip('\n') for name in f.readlines()]
    #         assert names[0] == 'input,gt,diff_gray,i_gray'
    #         for name in names[1:]:
    #             name = name.replace('rt_tif_16bit_540p', 'rt_8bit_1500p')
    #             name = name.replace('gt_16bit_540p', 'gt_8bit_1500p_only_adjust_light')
    #             name = name.replace('tif', 'jpg')
    #             arr = name.split(',')
    #             if not os.path.exists(os.path.join(root_path, arr[0])) or not os.path.exists(os.path.join(root_path, arr[1])):
    #                 continue
    #             key = arr[0].split('/')[-1]
    #             assert key not in remind_keys
    #             remind_keys.append(key)
    #             select_names_pair.append((arr[0], arr[1]))
    #             # print('{}-{}'.format(arr[0], arr[1]))
    #     return select_names_pair,  remind_keys
    #
    # select_names_pair, remind_keys = select_names()

    # print('select_names_pair = {}'.format(len(select_names_pair)))

    in_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.jpg'
    gt_txt = 'gt_8bit_1500p_only_adjust_light.txt'
    rt_txt = 'rt_8bit_1500p.txt'
    over_txt = 'rt_8bit_only_adjust_light_over_1500p.txt'

    gt_names = get_names(os.path.join(in_root_path, gt_txt))
    rt_names = get_names(os.path.join(in_root_path, rt_txt))
    over_names = get_names(os.path.join(in_root_path, over_txt))

    print('over = {}'.format(len(over_names)))

    over_keys_names = dict([(name.split('/')[-1], name) for name in over_names])
    gt_keys_names = dict([(name.split('/')[-1], name) for name in gt_names])
    rt_keys_names = dict([(name.split('/')[-1], name) for name in rt_names])

    names_pair = list()
    # for key, name in over_keys_names.items():
    #     names_pair.append((name, gt_keys_names[key]))
    #
    # print('over pair {}'.format(len(names_pair)))

    # names_pair = random.sample(select_names_pair, k=int(len(select_names_pair)*0.5))
    # print('select pair {}'.format(len(names_pair)))

    # remind_keys.extend(over_keys_names.keys())

    index = 0
    rt_names_pair = list()
    for key, name in rt_keys_names.items():
        if key in over_keys_names.keys():
            index += 1
            continue
        if not os.path.exists(os.path.join(in_root_path, name)) or not os.path.exists(os.path.join(in_root_path, gt_keys_names[key])):
            continue
        rt_names_pair.append((name, gt_keys_names[key]))
    
    print('rt pair {}, rt {}'.format(len(rt_names_pair), len(rt_keys_names)))
    
    # index = [i for i in range(len(rt_names_pair))]
    # random.shuffle(rt_names_pair)
    # rt_names_pair = random.sample(rt_names_pair, k=int(len(names_pair)*2))
    print('select rt {}'.format(len(rt_names_pair)))
    names_pair.extend(rt_names_pair)
    print('total pair {}'.format(len(names_pair)))

    out_root_path = '/home/shengdewu/data/xt.image.enhancement.540.jpg'
    os.makedirs(out_root_path, exist_ok=True)

    random.shuffle(names_pair)
    test_nums = int(0.05 * len(names_pair))
    test_names = names_pair[: test_nums]
    train_names = names_pair[test_nums: ]
    train_point = int(0.5 * len(train_names))
    print('train {}, test {}'.format(len(train_names), len(test_names)))
    with open(os.path.join(out_root_path, 'only_light.test.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in test_names:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'only_light.train_input.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[: train_point]:
            f.write('{},{}\n'.format(name[0], name[1]))

    with open(os.path.join(out_root_path, 'only_light.train_label.txt'), mode='w') as f:
        f.write('input,gt\n')
        for name in train_names[train_point :]:
            f.write('{},{}\n'.format(name[0], name[1]))

    tmp_root_path = '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540.jpg.tmp'
    os.makedirs(os.path.join(tmp_root_path, 'gt_8bit_1500p_only_adjust_light'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_8bit_1500p'), exist_ok=True)
    os.makedirs(os.path.join(tmp_root_path, 'rt_8bit_only_adjust_light_over_1500p'), exist_ok=True)

    os.makedirs(os.path.join(out_root_path, 'gt_8bit_1500p_only_adjust_light'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_8bit_1500p'), exist_ok=True)
    os.makedirs(os.path.join(out_root_path, 'rt_8bit_only_adjust_light_over_1500p'), exist_ok=True)

    for name in tqdm.tqdm(names_pair):
        # print('{} -> {}'.format(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0])))
        # print('{} -> {}'.format(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1])))
        if os.path.exists(os.path.join(out_root_path, name[0])) and os.path.exists(os.path.join(out_root_path, name[1])):
            continue

        shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(out_root_path, name[0]))
        shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(out_root_path, name[1]))
        if random.random() <= 0.01:
            shutil.copy(os.path.join(in_root_path, name[0]), os.path.join(tmp_root_path, name[0]))
            shutil.copy(os.path.join(in_root_path, name[1]), os.path.join(tmp_root_path, name[1]))            
    return


if __name__ == '__main__':
    # require_txt = [
    #     'no_aug.train_input.txt',
    #     'no_aug.train_label.txt',
    #     'no_aug.test.txt'
    # ]
    # gray_statistics = GrayDataSet(require_txt, '/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540')
    # gray_statistics()

    # select_equal_brightness()
    #
    # add_over_expose()

    # bright_rt_img()

    # show_im_img()

    # create_label()
    # check()
    # cp()
    # select_img()
    # class_img()
    # select_class_img()
    # add_over_expose_by_gt()
    # add_under_expose_by_gt()
    # split_over_expose_by_gt()
    # split_only_over_expose_by_gt()
    split_only_over_under_expose_by_gt()
    # add_over_expose_by_gt_jpg()
    # split_over_expose_by_gt_jpg()


