from engine.config.parser import default_argument_parser
from engine.config import get_cfg
import os
from inference.eval import Inference, compare
import torch
import shutil
from fvcore.common.config import CfgNode


if __name__ == '__main__':

    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    weights = cfg.MODEL.WEIGHTS
    model_path_root = weights[0: weights.rfind('/AdaptivePairedModel')]

    if os.path.exists(os.path.join(model_path_root, 'config.yaml')):
        print('use {}'.format(os.path.join(model_path_root, 'config.yaml')))

        f = open(os.path.join(model_path_root, 'config.yaml'), mode='r')
        hcfg = CfgNode().load_cfg(f)
        f.close()
        cfg.SOLVER.LAMBDA_PIXEL = hcfg.SOLVER.LAMBDA_PIXEL
        cfg.SOLVER.LAMBDA_SMOOTH = hcfg.SOLVER.LAMBDA_SMOOTH
        cfg.SOLVER.LAMBDA_MONOTONICITY = hcfg.SOLVER.LAMBDA_MONOTONICITY
        # cfg.MODEL.VGG.VGG_LAYER = hcfg.MODEL.VGG.VGG_LAYER
        cfg.MODEL.ARCH = hcfg.MODEL.ARCH
        cfg.MODEL.LUT.SUPPLEMENT_NUMS = hcfg.MODEL.LUT.SUPPLEMENT_NUMS
        cfg.MODEL.LUT.DIMS = hcfg.MODEL.LUT.DIMS
        cfg.MODEL.LUT.ZERO_LUT = hcfg.MODEL.LUT.ZERO_LUT
        cfg.MODEL.CLASSIFIER.ARCH = hcfg.MODEL.CLASSIFIER.ARCH
        cfg.MODEL.CLASSIFIER.RESNET_ARCH = hcfg.MODEL.CLASSIFIER.RESNET_ARCH
    cfg.freeze()

    eval = Inference(cfg)
    eval.resume_or_load()
    eval.loop()

    torch.cuda.empty_cache()

    root_path = '/home/shengdewu/data_shadow/train.output/imagelut.test'
    out_root = '/home/shengdewu/data_shadow/train.output/imagelut.test'

    base_path = os.path.join(root_path, 'base')

    compare_name = ['imagelut.c18.m20.5e4', 'imagelut.c18.p5.m20.5e4']
    compare_paths = [os.path.join(root_path, name) for name in compare_name]
    out_path = os.path.join(out_root, 'compare-{}'.format('-'.join(compare_name)))
    compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=False)

    rhd = open('./error.txt', mode='r')
    error = [line.strip('\n') for line in rhd.readlines()]
    rhd.close()

    os.makedirs(os.path.join(out_path, 'check'), exist_ok=True)
    for name in error:
        if not os.path.exists(os.path.join(out_path, '{}.tif'.format(name))):
            continue
        shutil.copy2(os.path.join(out_path, '{}.tif'.format(name)), os.path.join(out_path, 'check', '{}.tif'.format(name)))
