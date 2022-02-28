import torch
import os
import tqdm
from inference.eval import Inference
from dataloader.build import build_dataset
import logging


class InferenceNoneGt(Inference):
    def __init__(self, cfg, tif=True):
        super(InferenceNoneGt, self).__init__(cfg, tif)
        return

    def loop(self, cfg, skip=False, special_name=None):
        if special_name is not None:
            assert (isinstance(special_name, list) or isinstance(special_name, tuple)) and len(special_name) > 0

        output = cfg.OUTPUT_DIR
        os.makedirs(output, exist_ok=True)
        test_dataset = build_dataset(cfg, model='test')
        logging.getLogger(__name__).info('create dataset {}, load {} test data'.format(cfg.DATALOADER.DATASET, len(test_dataset)))

        img_format = 'jpg' if self.unnormalizing_value == 255 else 'tif'
        skin_name = list()
        if skip:
            skin_name = [name for name in os.listdir(output) if name.lower().endswith(img_format)]

        for index in tqdm.tqdm(range(len(test_dataset))):
            data = test_dataset.get_item(index, skin_name, special_name=special_name, img_format=img_format)
            if data['A_input'] is None:
                continue
            # data = self.test_dataset[index]
            input_name = data['input_name']
            # if input_name.endswith('tif') and img_format != 'tif':
            #     input_name = '{}.{}'.format(input_name[:input_name.rfind('.tif')], img_format)
            #
            # if special_name is not None and input_name not in special_name:
            #     continue
            # if input_name in skin_name:
            #     continue
            real_A = data["A_input"].to(self.device).unsqueeze(0)

            combine_lut = self.model.generate_lut(real_A)
            _, fake_B = self.triliear(combine_lut, real_A)

            img_sample = torch.cat((real_A, fake_B), -1)

            Inference.save_image(img_sample, '{}/{}'.format(output, input_name), unnormalizing_value=self.unnormalizing_value, nrow=1, normalize=False)
        return


