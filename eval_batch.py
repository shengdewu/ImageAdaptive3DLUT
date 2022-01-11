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

    model_path_root = cfg.MODEL.WEIGHTS
    out_root = cfg.OUTPUT_DIR
    assert model_path_root != out_root

    cfg.defrost()
    handle = open('imglut.txt', mode='r')
    model_path_names = [line.strip('\n') for line in handle.readlines()]
    handle.close()
    for model_path in model_path_names:
        cfg.OUTPUT_DIR = os.path.join(out_root, model_path)
        if os.path.exists(cfg.OUTPUT_DIR):
            print('skip {}'.format(cfg.OUTPUT_DIR))
            continue

        path = os.path.join(model_path_root, model_path, 'last_checkpoint')
        rh = open(path, mode='r')
        check_name = rh.readline().strip('\n')

        assert os.path.exists(os.path.join(model_path_root, model_path, check_name)), 'the {} is not exists'.format(os.path.join(model_path_root, model_path, check_name))
        cfg.MODEL.WEIGHTS = os.path.join(model_path_root, model_path, check_name)

        if os.path.exists(os.path.join(model_path_root, model_path, 'config.yaml')):
            f = open(os.path.join(model_path_root, model_path, 'config.yaml'), mode='r')
            hcfg = CfgNode().load_cfg(f)
            f.close()
            cfg.MODEL.LUT.SUPPLEMENT_NUMS = hcfg.MODEL.LUT.SUPPLEMENT_NUMS
        else:
            lut = model_path.split('.')[1]
            assert lut.find('c') != -1, 'this lut is not require {}'.format(lut)
            cfg.MODEL.LUT.SUPPLEMENT_NUMS = int(lut[1:])

        print('use {} out {} lut {}'.format(cfg.MODEL.WEIGHTS, cfg.OUTPUT_DIR, cfg.MODEL.LUT.SUPPLEMENT_NUMS))

        eval = Inference(cfg)
        eval.resume_or_load()
        eval.loop()

        torch.cuda.empty_cache()

    root_path = '/home/shengdewu/data_shadow/train.output/imagelut.test'
    out_path = '/home/shengdewu/data_shadow/train.output/imagelut.test'

    base_path = os.path.join(root_path, 'base')
    rhand = open('imglut.output.txt', mode='r')
    img_path_names = [line.strip('\n') for line in rhand.readlines()]

    rhd = open('./error.txt', mode='r')
    error = [line.strip('\n') for line in rhd.readlines()]
    rhd.close()

    for img_path_name in img_path_names:
        if img_path_name in ['imagelut.c18.m20', 'imagelut.c18', 'imagelut.c18.m25.5e-4', 'imagelut.c18.1e-3', 'imagelut.c18.1e-5', 'imagelut.c18.1e-6', 'imagelut.c18.m5']:
            continue

        compare_paths = [os.path.join(root_path, 'imagelut.c18.m20'), os.path.join(root_path, img_path_name)]

        op = os.path.join(out_path, 'compare.base-{}-{}'.format('imagelut.c18.m20', img_path_name))
        if os.path.exists(op):
            print('skip {} {}'.format(img_path_name, op))
            continue

        print('compare {} {}'.format(base_path, compare_paths))

        compare(base_path=base_path, compare_paths=compare_paths, out_path=op)

        os.makedirs(os.path.join(op, 'check'), exist_ok=True)
        for name in error:
            if not os.path.exists(os.path.join(op, '{}.tif'.format(name))):
                continue
            shutil.copy2(os.path.join(op, '{}.tif'.format(name)), os.path.join(op, 'check', '{}.tif'.format(name)))
