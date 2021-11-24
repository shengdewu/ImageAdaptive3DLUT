import torch.utils.data
from dataloader.datasets_rgb import ImageDataset_sRGB, ImageDataset_sRGB_unpaired
from dataloader.datasets_xyz import ImageDataset_XYZ, ImageDataset_XYZ_unpaired
from engine.samplers.distributed_sampler import TrainingSampler
from engine.data.common import ToIterableDataset
import multiprocessing


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
        imgA_input = list()
        imgA_exptC = list()
        imgB_exptC = list()
        input_name = list()
        for data in batch_img_list:
            imgA_input.append(data['A_input'])
            imgA_exptC.append(data['A_exptC'])
            if 'B_exptC' in data.keys():
                imgB_exptC.append(data['B_exptC'])
            input_name.append(data['input_name'])

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in imgA_input]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values.numpy().tolist()

        input_batch = dict()
        # max_size can be a tensor in tracing mode, therefore convert to list
        input_batch['A_input'] = DataLoader.pad_img(imgA_input, max_size)
        input_batch['A_exptC'] = DataLoader.pad_img(imgA_exptC, max_size)
        if len(imgB_exptC) > 0:
            input_batch['B_exptC'] = DataLoader.pad_img(imgB_exptC, max_size)
        input_batch['input_name'] = input_name

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
        if cfg.INPUT.COLOR_SPACE == 'sRGB':
            if cfg.INPUT.UNPAIRED:
                train_dataset = ImageDataset_sRGB_unpaired(cfg.DATALOADER.ROOT_PAT, mode="train")
                test_dataset = ImageDataset_sRGB_unpaired(cfg.DATALOADER.ROOT_PAT, mode="test")
            else:
                train_dataset = ImageDataset_sRGB(cfg.DATALOADER.ROOT_PAT, mode="train")
                test_dataset = ImageDataset_sRGB(cfg.DATALOADER.ROOT_PAT, mode="test")

        elif cfg.INPUT.COLOR_SPACE == 'XYZ':

            if cfg.INPUT.UNPAIRED:
                train_dataset = ImageDataset_XYZ_unpaired(cfg.DATALOADER.ROOT_PAT, mode="train")
                test_dataset = ImageDataset_XYZ_unpaired(cfg.DATALOADER.ROOT_PAT, mode="test")
            else:
                train_dataset = ImageDataset_XYZ(cfg.DATALOADER.ROOT_PAT, mode="train")
                test_dataset = ImageDataset_XYZ(cfg.DATALOADER.ROOT_PAT, mode="test")

        else:
            raise NotImplemented(cfg.INPUT.COLOR_SPACE)
        return train_dataset, test_dataset

    @staticmethod
    def create_dataloader(dataset, num_workers=1, batch_size=1):
        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=DataLoader.collate_fn
        )

    @staticmethod
    def create_sampler_dataloader(dataset, batch_size, num_workers):
        sampler = TrainingSampler(len(dataset))

        dataset = ToIterableDataset(dataset, sampler)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True)

    @staticmethod
    def create_distribute_sampler_dataloder(dataset, batch_size, rank, world_size, num_workers):
        sampler = TrainingSampler(len(dataset), rank=rank, world_size=world_size)

        dataset = ToIterableDataset(dataset, sampler)

        max_workers = multiprocessing.cpu_count()
        num_workers = num_workers if num_workers < max_workers else max_workers

        return torch.utils.data.DataLoader(
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=DataLoader.collate_fn,
                                    drop_last=True)
