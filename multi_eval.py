import os
from inference.eval import compare
from eval import merge_config, convert2jpg
from inference.test import InferenceNoneGt
import torch
import random


if __name__ == '__main__':
    rhd = open('./dir.names.txt', mode='r')
    dir_names = [line.strip('\n') for line in rhd.readlines()]
    rhd.close()

    cfg = merge_config()
    cfg.defrost()
    eval = InferenceNoneGt(cfg, tif=False)
    eval.resume_or_load()

    torch.cuda.empty_cache()
    out_root = cfg.OUTPUT_DIR
    data_path = cfg.DATALOADER.DATA_PATH
    compare_root = out_root[:out_root.rfind('/')]
    compare_base = os.path.join(compare_root, 'online.test.base')
    for dir_name in dir_names:
        data_cfg = cfg.clone()
        out_path = os.path.join(out_root, dir_name)
        data_cfg.OUTPUT_DIR = out_path
        data_cfg.DATALOADER.DATA_PATH = os.path.join(data_path, dir_name)
        print('use {}'.format(data_cfg.DATALOADER.DATA_PATH))

        eval.loop(data_cfg, skip=True)

        base_path = os.path.join(compare_base, dir_name)
        convert2jpg(base_path)

        compare_name = ['imagelut.c12.m30.s1.c5e5.p6.vgg.0.05.0159999', 'imagelut.c12.m10.s1.c1e4.p6.vgg.0.5.0249999', 'imagelut.c12.m10.1e4.vgg.rin.p6.139999']
        #compare_name = ['imagelut.c12.m10.s1.c1e4.p6.vgg.0.5.0249999', 'imagelut.c19.m30.s1e3.c1e4.p1.vgg1e4.aug.0229999', 'imagelut.c19.m30.s1e3.c1e4.p1.vgg1e4.aug.0349999']
        #compare_name = ['imagelut.c19.m30.s1e3.c1e4.p1.vgg1e4.aug.0229999', 'imagelut.c19.m30.s1e3.c1e4.p1.vgg1e4.aug.0349999']
        compare_paths = [os.path.join(compare_root, name, dir_name) for name in compare_name]
        out_path = os.path.join(compare_root, 'online.test.all.compare-{}'.format('-'.join(compare_name)), dir_name)
        special_name = [name for name in os.listdir(base_path) if name.lower().endswith('jpg')]
        compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=True, special_name=special_name)

