from fvcore.common.registry import Registry  # for backward compatibility.
import logging


TRAINER_ARCH_REGISTRY = Registry("NETWORK")
TRAINER_ARCH_REGISTRY.__doc__ = """

Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
from.trainer_unpaired import TrainerUnPaired
from.trainer_paired import TrainerPaired


def build_trainer(cfg):
    model = TRAINER_ARCH_REGISTRY.get(cfg.trainer)(cfg)
    logging.info("select {} as network model".format(cfg.trainer))
    return model
