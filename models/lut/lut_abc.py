import torch
from typing import Any, Callable


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
