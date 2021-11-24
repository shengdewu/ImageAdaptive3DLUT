from fvcore.common.registry import Registry  # for backward compatibility.
import logging

CLASSIFIER_ARCH_REGISTRY = Registry("NETWORK")
CLASSIFIER_ARCH_REGISTRY.__doc__ = """

Registry for classifier, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

from .classifier import Classifier
from .classifier import ClassifierUnpaired
from .classifier import ClassifierResnet


def build_classifier(cfg):
    model = CLASSIFIER_ARCH_REGISTRY.get(cfg.MODEL.CLASSIFIER.ARCH,)(cfg)
    return model
