import torch
from typing import Any, Callable
import numpy as np


def _forward_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


class LutAbc(torch.nn.Module):
    def __init__(self):
        super(LutAbc, self).__init__()
        self._lut = None
        return

    forward: Callable[..., Any] = _forward_unimplemented

    @property
    def lut(self):
        return self._lut

    def transfer2cube(self, size):
        lut3d = self._lut.detach().cpu().numpy()
        n, dim1, dim2, dim3 = lut3d.shape
        assert dim1 == dim2 and dim1 == dim3
        assert size % dim1 == 0
        box = int(size / dim1)
        assert box * box == dim1, 'the box power({}, 2) must be == {}'.format(box, dim1)
        lut2d = np.zeros((size, size, 3), dtype=np.float32)
        self.transfer(box, dim1, lut3d, lut2d)
        return lut2d

    def transfer(self, box, dim, lut3d, lut2d):
        for bx in range(box):
            for by in range(box):
                for g in range(dim):
                    for r in range(dim):
                        b = bx + by * box
                        x = r + bx * dim
                        y = g + by * dim
                        lut2d[y, x, 0] = lut3d[0, b, g, r]
                        lut2d[y, x, 1] = lut3d[1, b, g, r]
                        lut2d[y, x, 2] = lut3d[2, b, g, r]
        return np.clip(lut2d, 0.0, 1.0)[:, :, ::-1]
