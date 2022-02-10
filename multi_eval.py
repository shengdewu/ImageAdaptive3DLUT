import os
from inference.eval import compare
from eval import merge_config, convert2jpg
from inference.test import InferenceNoneGt
import torch


def inference(cfg, special_name=None):
    eval = InferenceNoneGt(cfg, tif=False)
    eval.resume_or_load()
    eval.loop(skip=True, special_name=special_name)

    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    rhd = open('./dir.names.txt', mode='r')
    dir_names = [line.strip('\n') for line in rhd.readlines()]
    rhd.close()

    cfg = merge_config()
    cfg.defrost()
    out_root = cfg.OUTPUT_DIR
    data_path = cfg.DATALOADER.DATA_PATH
    compare_root = out_root[:out_root.rfind('/')]
    compare_base = os.path.join(compare_root, 'online.test.base')
    for dir_name in dir_names:
        out_path = os.path.join(out_root, dir_name)
        cfg.OUTPUT_DIR = out_path
        cfg.DATALOADER.DATA_PATH = os.path.join(data_path, dir_name)

        inference(cfg)

        base_path = os.path.join(compare_base, dir_name)
        convert2jpg(base_path)

        compare_name = ['imagelut.c12.m20.1e4.vgg.rin.p6.online.test']
        compare_paths = [os.path.join(compare_root, name, dir_name) for name in compare_name]
        out_path = os.path.join(compare_root, 'online.test.compare-{}'.format('-'.join(compare_name)), dir_name)
        special_name = [name for name in os.listdir(base_path) if name.lower().endswith('jpg')]
        compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=True, special_name=special_name)

