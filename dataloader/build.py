from fvcore.common.registry import Registry  # for backward compatibility.

DATASET_ARCH_REGISTRY = Registry("DATA")
DATASET_ARCH_REGISTRY.__doc__ = """

Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_dataset(cfg, model):
    dataset = DATASET_ARCH_REGISTRY.get(cfg.DATALOADER.DATASET)(cfg, model)
    return dataset
