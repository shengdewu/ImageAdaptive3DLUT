import torch
import numpy as np
import os
from .lut_abc import LutAbc
from trilinear.TrilinearInterpolationModel import TrilinearInterpolationModel

__all__ = [
    'Generator_3DLUT_identity',
    'Generator_3DLUT_n_zero'
]


class Generator_3DLUT_identity(LutAbc):
    def __init__(self, dim=33, device='cuda'):
        super(Generator_3DLUT_identity, self).__init__()

        lines = self.__load_identify_lut(dim)

        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])

        self._lut = torch.nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.tilinear_interpolation = TrilinearInterpolationModel()
        self.to(device)
        return

    @staticmethod
    def __load_identify_lut(dim):
        dir_name = os.path.dirname(__file__)
        if dim == 33:
            file = open(os.path.join(dir_name, 'IdentityLUT33.txt'), 'r')
        elif dim == 64:
            file = open(os.path.join(dir_name, 'IdentityLUT64.txt'), 'r')
        else:
            raise FileNotFoundError('not found identify lut for dim {}'.format(dim))
        return file.readlines()

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

    def state_dict(self, offset=0):
        state_dict = dict()
        for i, lut in self.generator_3d_lut.items():
            state_dict[i+offset] = lut.state_dict()
        return state_dict

    def load_state_dict(self, state_dict:dict, offset=1):
        for i, lut in self.generator_3d_lut.items():
            lut.load_state_dict(state_dict[i+offset])
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

    def __len__(self):
        return len(self.generator_3d_lut)

    def __call__(self, x):
        output = dict()
        for i, lut in self.generator_3d_lut.items():
            output[i] = lut(x)
        return output
