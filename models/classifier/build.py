from fvcore.common.registry import Registry  # for backward compatibility.


CLASSIFIER_ARCH_REGISTRY = Registry("NETWORK")
CLASSIFIER_ARCH_REGISTRY.__doc__ = """

Registry for classifier, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_classifier(cfg):
    model = CLASSIFIER_ARCH_REGISTRY.get(cfg.MODEL.CLASSIFIER.ARCH,)(cfg)
    return model
