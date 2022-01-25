from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import os
from inference.eval import Inference, compare
import torch
import shutil
from fvcore.common.config import CfgNode
import re


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

    train_config = os.path.join(model_path_root, 'config.yaml')
    if os.path.exists(train_config):
        print('use {}'.format(train_config))

        f = open(train_config, mode='r')
        hcfg = CfgNode().load_cfg(f)
        f.close()

        VGG_PATH = cfg.MODEL.VGG.VGG_PATH
        WEIGHTS = cfg.MODEL.WEIGHTS
        PRETRAINED_PATH = cfg.MODEL.CLASSIFIER.PRETRAINED_PATH
        device = cfg.MODEL.DEVICE

        cfg.SOLVER = hcfg.SOLVER
        cfg.MODEL = hcfg.MODEL

        cfg.MODEL.VGG.VGG_PATH = VGG_PATH
        cfg.MODEL.WEIGHTS = WEIGHTS
        cfg.MODEL.CLASSIFIER.PRETRAINED_PATH = PRETRAINED_PATH
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def inference(cfg, special_name=None):
    eval = Inference(cfg)
    eval.resume_or_load()
    eval.loop(skip=False, special_name=special_name)

    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    rhd = open('./error.txt', mode='r')
    error_names = ['{}.tif'.format(line.strip('\n')) for line in rhd.readlines()]
    rhd.close()

    cfg = merge_config()
    inference(cfg, error_names)

    root_path = cfg.OUTPUT_DIR
    root_path = root_path[:root_path.rfind('/')]

    out_root = '/home/shengdewu/data_shadow/train.output/imagelut.test'

    base_path = os.path.join(root_path, 'base')

    compare_name = ['imagelut.c18.m20.5e4', 'imagelut.c12.m10.1e4.vgg.rin.p2']
    compare_paths = [os.path.join(root_path, name) for name in compare_name]
    out_path = os.path.join(out_root, 'compare-{}'.format('-'.join(compare_name)))
    compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=False)

    os.makedirs(os.path.join(out_path, 'check'), exist_ok=True)
    for name in error_names:
        if not os.path.exists(os.path.join(out_path, name)):
            continue
        shutil.copy2(os.path.join(out_path, name), os.path.join(out_path, 'check', name))
