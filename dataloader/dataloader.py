import torch.utils.data
from dataloader.datasets_rgb import ImageDataset_sRGB, ImageDataset_sRGB_unpaired
from dataloader.datasets_xyz import ImageDataset_XYZ, ImageDataset_XYZ_unpaired


class DataLoader:
    def __init__(self):
        return

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
        )

        psnr_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        return dataloader, psnr_dataloader, test_dataset
