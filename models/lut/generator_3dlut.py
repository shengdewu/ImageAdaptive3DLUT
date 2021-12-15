import torch
import numpy as np
from engine.functional import get_model_state_dict, load_model_state_dict
from .lut_abc import LutAbc
from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel

__all__ = [
    'Generator_3DLUT_identity',
    'Generator_3DLUT_n_zero'
]


class Generator_3DLUT_identity(LutAbc):
    def __init__(self, dim=33, device='cuda'):
        super(Generator_3DLUT_identity, self).__init__()

        buffer = self.generate_identity_lut(dim)

        self._lut = torch.nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.tilinear_interpolation = TrilinearInterpolationModel()
        self.to(device)
        return

    @staticmethod
    def generate_identity_lut(dim):
        lut3d = np.zeros((3, dim, dim, dim), dtype=np.float32)
        step = 1.0 / float(dim-1)
        for b in range(dim):
            for g in range(dim):
                for r in range(dim):
                    lut3d[0, b, g, r] = step * r
                    lut3d[1, b, g, r] = step * g
                    lut3d[2, b, g, r] = step * b
        return lut3d

    def forward(self, x):
        _, output = self.tilinear_interpolation(self._lut, x)
        return output


class Generator_3DLUT_zero(LutAbc):
    def __init__(self, dim=33, device='cuda'):
        super(Generator_3DLUT_zero, self).__init__()

        self._lut = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self._lut = torch.nn.Parameter(torch.tensor(self._lut))
        self.trilinear_interpolation = TrilinearInterpolationModel()
        self.to(device)
        return

    def forward(self, x):
        _, output = self.trilinear_interpolation(self._lut, x)

        return output


class Generator_3DLUT_n_zero:
    def __init__(self, dim=33, nums=2, device='cuda'):
        self.generator_3d_lut = dict()  # index, lutmodel
        for i in range(nums):
            self.generator_3d_lut[i] = Generator_3DLUT_zero(dim, device)
        return

    def parameters(self):
        parameters = list()
        for i, lut in self.generator_3d_lut.items():
            parameters.append(lut.parameters())
        return parameters

    @staticmethod
    def __get_model_state_dict(model):
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            return model.module.state_dict()
        return model.state_dict()

    @staticmethod
    def __load_model_state_dict(model, state_dict):
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    def state_dict(self, offset=1):
        state_dict = dict()
        for i, lut in self.generator_3d_lut.items():
            state_dict[i+offset] = get_model_state_dict(lut)
        return state_dict

    def load_state_dict(self, state_dict:dict, offset=1):
        total_lut = len([key for key in state_dict.keys() if key >= offset])
        assert total_lut == len(self.generator_3d_lut), 'Generator_3DLUT_n_zero owned is not equal to that in the state_dict'
        for i, lut in self.generator_3d_lut.items():
            load_model_state_dict(lut, state_dict[i+offset])
        return

    def enable_parallel(self):
        parallel = dict()
        for i, lut in self.generator_3d_lut.items():
            parallel[i] = torch.nn.parallel.DataParallel(lut)
        self.generator_3d_lut.clear()
        self.generator_3d_lut.update(parallel)
        return

    def enable_model_distributed(self, gpu_id):
        parallel = dict()
        for i, lut in self.generator_3d_lut.items():
            lut = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lut)
            parallel[i] = torch.nn.parallel.DistributedDataParallel(lut, device_ids=[gpu_id])
        self.generator_3d_lut.clear()
        self.generator_3d_lut.update(parallel)
        return

    def train(self):
        for i, lut in self.generator_3d_lut.items():
            lut.train()
        return

    def eval(self):
        for i, lut in self.generator_3d_lut.items():
            lut.eval()
        return

    def tv(self, tv_model):
        tv = list()
        mn = list()
        for k, lut in self.generator_3d_lut.items():
            tv1, mn1 = tv_model(lut)
            tv.append(tv1)
            mn.append(mn1)
        return tv, mn

    def foreach(self):
        for k, lut in self.generator_3d_lut.items():
            yield k, lut
        return

    def upate(self, luts):
        self.generator_3d_lut.clear()
        for k, lut in luts.items():
            self.generator_3d_lut[k] = lut
        return

    def __len__(self):
        return len(self.generator_3d_lut)

    def __call__(self, x):
        output = dict()
        for i, lut in self.generator_3d_lut.items():
            output[i] = lut(x)
        return output
