import torch.utils.data
from dataloader.datasets_rgb import ImageDataset_sRGB, ImageDataset_sRGB_unpaired
from dataloader.datasets_xyz import ImageDataset_XYZ, ImageDataset_XYZ_unpaired


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
    def create_dataloader(cfg):
        if cfg.input_color_space == 'sRGB':
            if cfg.unpaired:
                train_dataset = ImageDataset_sRGB_unpaired(cfg.data_path, mode="train")
                test_dataset = ImageDataset_sRGB_unpaired(cfg.data_path, mode="test")
            else:
                train_dataset = ImageDataset_sRGB(cfg.data_path, mode="train")
                test_dataset = ImageDataset_sRGB(cfg.data_path, mode="test")

        elif cfg.input_color_space == 'XYZ':

            if cfg.unpaired:
                train_dataset = ImageDataset_XYZ_unpaired(cfg.data_path, mode="train")
                test_dataset = ImageDataset_XYZ_unpaired(cfg.data_path, mode="test")
            else:
                train_dataset = ImageDataset_XYZ(cfg.data_path, mode="train")
                test_dataset = ImageDataset_XYZ(cfg.data_path, mode="test")

        else:
            raise NotImplemented(cfg.input_color_space)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.n_cpu,
            collate_fn=DataLoader.collate_fn
        )

        psnr_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=DataLoader.collate_fn
        )

        return dataloader, psnr_dataloader, test_dataset
