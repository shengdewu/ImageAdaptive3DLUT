from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import os
from inference.eval import Inference, compare
import torch
import shutil
from fvcore.common.config import CfgNode
import re
import cv2
import numpy as np
from engine.log.logger import setup_logger
import engine.comm as comm
import logging


def merge_config():
    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    weights = cfg.MODEL.WEIGHTS
    pattern = re.compile(r'Adaptive.*\.pth')
    dg = pattern.search(weights)
    model_path_root = weights[: dg.span()[0]]

    setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=cfg.OUTPUT_LOG_NAME)

    train_config = os.path.join(model_path_root, 'config.yaml')
    rough_size = None
    down_factor = 1
    if os.path.exists(train_config):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('use {}'.format(train_config))

        f = open(train_config, mode='r')
        hcfg = CfgNode().load_cfg(f)
        f.close()

        vgg_path = cfg.MODEL.get('VGG', None)
        if vgg_path is not None:
            vgg_path = vgg_path.get('VGG_PATH', None)
        weight_path = cfg.MODEL.WEIGHTS
        classifier = cfg.MODEL.get('CLASSIFIER', None)
        if classifier is not None:
            rough_size = classifier.get('ROUGH_SIZE', None)
            if rough_size is None:
                down_factor = classifier.get('DOWN_FACTOR', 1)

        device = cfg.MODEL.DEVICE
        cfg.SOLVER = hcfg.SOLVER
        cfg.MODEL = hcfg.MODEL
        cfg.MODEL.CLASSIFIER.ROUGH_SIZE = None
        cfg.MODEL.CLASSIFIER.DOWN_FACTOR = 1
        if vgg_path is not None:
            cfg.MODEL.VGG.VGG_PATH = vgg_path
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.DEVICE = device
    cfg.freeze()

    path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")

    with open(path, "w") as f:
        f.write(cfg.dump(allow_unicode=True))

    return cfg, rough_size, down_factor


def inference(cfg, special_name=None):
    eval = Inference(cfg, tif=False)
    eval.resume_or_load()
    eval.loop(cfg=cfg, skip=True, special_name=special_name)

    torch.cuda.empty_cache()
    return


def convert2jpg(path, special_name=None):
    base_names = [name for name in os.listdir(path) if name.find('lut') == -1 and name.lower().endswith('tif')]
    skip_names = [name for name in os.listdir(path) if name.lower().endswith('jpg')]
    for name in base_names:
        if special_name is not None and name not in special_name:
            continue
        new_name = '{}.jpg'.format(name[:name.rfind('.tif')])
        if new_name in skip_names:
            continue
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_UNCHANGED)
        img = np.clip((img / 65535) * 255 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite('{}/{}'.format(path, new_name), img)


if __name__ == '__main__':
    rhd = open('./error.txt', mode='r')
    error_names = ['{}.jpg'.format(line.strip('\n')) for line in rhd.readlines()]
    rhd.close()

    cfg = merge_config()
    inference(cfg, special_name=None)

    root_path = cfg.OUTPUT_DIR
    root_path = root_path[:root_path.rfind('/')]

    base_path = os.path.join(root_path, 'base')
    rhd = open('./error.txt', mode='r')
    special_name = ['{}.tif'.format(line.strip('\n')) for line in rhd.readlines()]
    rhd.close()
    convert2jpg(base_path, special_name)
    special_name = [name for name in os.listdir(base_path) if name.lower().endswith('jpg')]

    compare_name = ['imagelut.c12.m10.1e4.vgg.rin.p6.0139999', 'imagelut.c12.m30.s1.c5e5.p6.vgg.0.05.0159999', 'imagelut.c12.m10.s1.c1e4.p6.vgg.0.5.0249999']
    compare_paths = [os.path.join(root_path, name) for name in compare_name]
    out_path = os.path.join(root_path, 'compare-{}'.format('-'.join(compare_name)))
    compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=True, special_name=special_name)

    os.makedirs(os.path.join(out_path, 'check'), exist_ok=True)
    for name in error_names:
        if not os.path.exists(os.path.join(out_path, name)):
            continue
        shutil.copy2(os.path.join(out_path, name), os.path.join(out_path, 'check', name))
