import torch
#from .python.TrilinearInterpolationFunction import TrilinearInterpolationFunction
from .cpp.TrilinearInterpolationFunction import TrilinearInterpolationFunction


class TrilinearInterpolationModel(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolationModel, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)
