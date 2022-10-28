import os
from inference.eval import compare
from eval import merge_config, convert2jpg
from inference.test import InferenceNoneGt
import torch
import random
import onnx_test.lut_onnx as lut_onnx
import compare_tool as compare_tool
import os


if __name__ == '__main__':
    rhd = open('/mnt/sda1/workspace/ImageAdaptive3DLUT/dir/error.txt', mode='r')
    dir_names = [line.strip('\n').split('#') for line in rhd.readlines()]
    rhd.close()

    cfg = merge_config()
    cfg.defrost()

    # onnx_path = '/mnt/sda1/workspace/ImageAdaptive3DLUT/onnx_test/lut16.onnx'
    # lut_onnx.to_onnx(cfg, onnx_path, input_size=(1, 3, 400, 400), log_name=cfg.OUTPUT_LOG_NAME)
    # mnn_path = '/mnt/sda1/workspace/ImageAdaptive3DLUT/onnx_test/lut16.mnn'
    # os.system('/home/shengdewu/workspace/MNN/build/MNNConvert -f ONNX --modelFile {} --MNNModel {} --bizCode biz'.format(onnx_path, mnn_path))
    # onnx_session = lut_onnx.load_onnx(onnx_path)

    model_eval = InferenceNoneGt(cfg, tif=False)
    model_eval.resume_or_load()

    torch.cuda.empty_cache()
    out_root = cfg.OUTPUT_DIR
    data_path = cfg.DATALOADER.DATA_PATH
    compare_root = out_root[:out_root.rfind('/')]
    compare_base = '/mnt/sdb/data.set/xintu.data/转档测评/线上样本'
    # compare_base = '/mnt/sdb/data.set/xintu.data/转档测评/人工样本'
    for dir_name in dir_names:
        save_dir_name = dir_name
        if isinstance(dir_name, list):
            save_dir_name = dir_name[0]
            dir_name = dir_name[1]
        data_cfg = cfg.clone()
        out_path = os.path.join(out_root, dir_name)
        data_cfg.OUTPUT_DIR = out_path
        data_cfg.DATALOADER.DATA_PATH = os.path.join(data_path, dir_name)
        print('use {}'.format(data_cfg.DATALOADER.DATA_PATH))

        model_eval.loop(data_cfg, skip=True, suppress_size=10)

        # ref_size = cfg.MODEL.CLASSIFIER.get('ROUGH_SIZE', None)
        # lut_onnx.onnx_run(down_factor=data_cfg.MODEL.CLASSIFIER.get('DOWN_FACTOR', 1), in_path=data_cfg.DATALOADER.DATA_PATH, out_path=data_cfg.OUTPUT_DIR, ort_session=onnx_session, ref_size=ref_size)

        base_path = os.path.join(compare_base, dir_name)

        compare_name = ['img.lut12.mobile.dim16.suppress', 'img.lut12.mobile.dim16.512', 'img.lut12.mobile.dim16.512.3']
        compare_paths = [os.path.join(compare_root, name, dir_name) for name in compare_name]
        out_path = os.path.join(compare_root, 'error.compare-{}'.format('-'.join(compare_name)), save_dir_name)
        special_name = [name for name in os.listdir(base_path) if name.lower().endswith('jpg')]
        if len(special_name) == 0:
            print('{} no item'.format(base_path))
            continue
        compare = compare_tool.CompareRow()
        compare.compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=False, special_name=None)

