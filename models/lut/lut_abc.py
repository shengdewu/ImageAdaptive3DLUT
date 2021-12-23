import torch
from typing import Any, Callable
import numpy as np


def transfer3d_2d(box, dim, lut3d, lut2d):
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
    return np.clip(lut2d, 0.0, 1.0)


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
