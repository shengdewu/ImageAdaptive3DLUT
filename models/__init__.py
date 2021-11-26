from fvcore.common.registry import Registry  # for backward compatibility.

MODEL_ARCH_REGISTRY = Registry("MODEL")
MODEL_ARCH_REGISTRY.__doc__ = """

Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

from .AdaptivePairedModel import AdaptivePairedModel
from .AdaptiveUnpairedModel import AdaptiveUnPairedModel


def build_model(cfg):
    model = MODEL_ARCH_REGISTRY.get(cfg.MODEL.ARCH)(cfg)
    return model
