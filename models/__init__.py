from fvcore.common.registry import Registry  # for backward compatibility.
import logging

CLASSIFIER_ARCH_REGISTRY = Registry("NETWORK")
CLASSIFIER_ARCH_REGISTRY.__doc__ = """

Registry for classifier, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

from .classifier.classifier import Classifier
from .classifier.classifier import ClassifierUnpaired
from .classifier.classifier import ClassifierResnet
from .classifier.classifier import ClassifierResnetUnpaired


def build_classifier(classifier, **cfg):
    model = CLASSIFIER_ARCH_REGISTRY.get(classifier)(**cfg)
    logging.info("select {} as classifier model".format(classifier))
    return model
