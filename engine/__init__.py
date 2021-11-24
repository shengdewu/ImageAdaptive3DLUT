import torch

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

__all__ = [k for k in globals().keys() if not k.startswith("_")]