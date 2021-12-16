import torch.utils.data
from dataloader.build import build_dataset
from engine.samplers.distributed_sampler import TrainingSampler
from engine.data.common import ToIterableDataset
import multiprocessing
import logging
import sys


class DataLoader:

    def __init__(self):
        return

    @staticmethod
    def pad_img(img_list, max_size):
        batch_shape = [len(img_list)] + list(img_list[0].shape[:-2]) + max_size
        img_batch = img_list[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(img_list, img_batch):
            sh = int((pad_img.shape[-2] - img.shape[-2]) / 2)
            sw = int((pad_img.shape[-1] - img.shape[-1]) / 2)
            pad_img[..., sh: img.shape[-2] + sh, sw: img.shape[-1] + sw].copy_(img)
        return img_batch

    @staticmethod
    def fromlist(batch_img_list):

        imgA_input = [data['A_input'] for data in batch_img_list]
        imgA_exptC = [data['A_exptC'] for data in batch_img_list]
        imgB_exptC = None
        if 'B_exptC' in batch_img_list[0].keys():
            imgB_exptC = [data['B_exptC'] for data in batch_img_list]

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in imgA_input] + [(im.shape[-2], im.shape[-1]) for im in imgA_exptC]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values.numpy().tolist()

        input_batch = dict()
        # max_size can be a tensor in tracing mode, therefore convert to list
        input_batch['A_input'] = DataLoader.pad_img(imgA_input, max_size)
        input_batch['A_exptC'] = DataLoader.pad_img(imgA_exptC, max_size)
        if imgB_exptC is not None and len(imgB_exptC) > 0:
            input_batch['B_exptC'] = DataLoader.pad_img(imgB_exptC, max_size)
        input_batch['input_name'] = [data['input_name'] for data in batch_img_list]

        # import numpy as np
        # import PIL.Image
        #
        # for i in range(len(input_batch['A_input'])):
        #     A_input = input_batch['A_input'][i]
        #     A_exptC = input_batch['A_exptC'][i]
        #     B_exptC = input_batch['B_exptC'][i]
        #     name = input_batch['input_name'][i]
        #
        #     A_input = PIL.Image.fromarray(np.uint8(np.array(A_input) * 255.0).transpose([1, 2, 0]))
        #     A_exptC = PIL.Image.fromarray(np.uint8(np.array(A_exptC) * 255.0).transpose([1, 2, 0]))
        #     B_exptC = PIL.Image.fromarray(np.uint8(np.array(B_exptC) * 255.0).transpose([1, 2, 0]))
        #
        #     w, h = A_input.size
        #     sh_img = PIL.Image.new(A_input.mode, (w*3, h))
        #     sh_img.paste(A_input, (0, 0))
        #     sh_img.paste(A_exptC, (w, 0))
        #     sh_img.paste(B_exptC, (w*2, 0))
        #
        #     sh_img.show(name)
        #     sh_img.close()

        return input_batch

    @staticmethod
    def collate_fn(batch_data):
        return DataLoader.fromlist(batch_data)

    @staticmethod
    def create_dataset(cfg):
        train_dataset = build_dataset(cfg, model='train')
        test_dataset = build_dataset(cfg, model='test')
        return train_dataset, test_dataset


    @staticmethod
    def create_sampler_dataloader(dataset, batch_size=1, num_workers=1, default_log_name=None):
        sampler = torch.utils.data.RandomSampler(dataset)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        if default_log_name is not None:
            logging.getLogger(default_log_name).info('{}: num workers={} batch size={}'.format(sys._getframe().f_code.co_name, num_workers, batch_size))

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True,
                                    sampler=sampler)

    @staticmethod
    def create_distribute_sampler_dataloder(dataset, batch_size, rank, world_size, num_workers, default_log_name=None):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        if default_log_name is not None:
            logging.getLogger(default_log_name).info('{}: num workers={} batch size={}'.format(sys._getframe().f_code.co_name, num_workers, batch_size))

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True,
                                    sampler=sampler)

    @staticmethod
    def create_sampler_iterable_dataloader(dataset, batch_size, num_workers, default_log_name=None):
        sampler = TrainingSampler(len(dataset))
        dataset = ToIterableDataset(dataset, sampler)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        if default_log_name is not None:
            logging.getLogger(default_log_name).info('{}: num workers={} batch size={}'.format(sys._getframe().f_code.co_name, num_workers, batch_size))

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True)

    @staticmethod
    def create_distribute_sampler_iterable_dataloder(dataset, batch_size, rank, world_size, num_workers, default_log_name=None):
        sampler = TrainingSampler(len(dataset), rank=rank, world_size=world_size)
        dataset = ToIterableDataset(dataset, sampler)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        if default_log_name is not None:
            logging.getLogger(default_log_name).info('{}: num workers={} batch size={}'.format(sys._getframe().f_code.co_name, num_workers, batch_size))

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True)
