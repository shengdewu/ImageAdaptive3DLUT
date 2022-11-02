import os
from eval import merge_config, convert2jpg
from inference.test import InferenceNoneGt
import torch
import random
import onnx_test.lut_onnx as lut_onnx
import compare_tool as compare_tool
import os


def inference(dir_names, onnx_path='', compare_in=None, compare_out=None):
    cfg = merge_config()
    cfg.defrost()

    if onnx_path != '':
        lut_onnx.to_onnx(cfg, onnx_path, input_size=(1, 3, 400, 400), log_name=cfg.OUTPUT_LOG_NAME)
        mnn_path = onnx_path.replace('.onnx', '.mnn')
        os.system(
            '/home/shengdewu/workspace/MNN/build/MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz'.format(
                onnx_path, mnn_path))
        session = lut_onnx.OnnxSession(onnx_path)
    else:
        session = InferenceNoneGt(cfg, tif=False)
        session.resume_or_load()

    torch.cuda.empty_cache()
    out_root = cfg.OUTPUT_DIR
    data_path = cfg.DATALOADER.DATA_PATH

    for dir_name in dir_names:
        save_dir_name = dir_name
        if isinstance(dir_name, list):
            save_dir_name = dir_name[0]
            dir_name = dir_name[1]
        data_cfg = cfg.clone()
        out_path = os.path.join(out_root, dir_name)
        data_cfg.OUTPUT_DIR = out_path
        data_cfg.DATALOADER.DATA_PATH = os.path.join(data_path, dir_name)
        print('enhance {}'.format(data_cfg.DATALOADER.DATA_PATH))

        session.loop(data_cfg, skip=True, suppress_size=10)

        if compare_in is not None:
            compare_paths = [os.path.join(compare_path, dir_name) for compare_path in compare_in]
            out_path = os.path.join(compare_out, save_dir_name)
            compare = compare_tool.CompareRow()
            compare.compare(base_path='', compare_paths=compare_paths, out_path=out_path, skip=True, special_name=None)
    return


if __name__ == '__main__':
    rhd = open('/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/dir/error.txt', mode='r')
    dir_names = [line.strip('\n').split('#') for line in rhd.readlines()]
    rhd.close()

    compare_paths = ['/mnt/sdb/data.set/xintu.data/转档测评/线上样本',
                     '/mnt/sda1/valid.output/enhance.test/img.lut12.mobile.dim16',
                     '/mnt/sda1/valid.output/enhance.test/img.lut12.mobile.dim16.512',
                     '/mnt/sda1/valid.output/enhance.test/img.lut12.mobile.dim16.512.3']
    onnx_path = '/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/onnx_test/lut16.onnx'
    compare_out = '/mnt/sda1/valid.output/enhance.test/compare-base-dim16-dim16.512-dim16.512.3'
    inference(dir_names, onnx_path, compare_paths, compare_out)
