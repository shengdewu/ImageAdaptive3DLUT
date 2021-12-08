from fvcore.common.registry import Registry  # for backward compatibility.

DATASET_ARCH_REGISTRY = Registry("DATA")
DATASET_ARCH_REGISTRY.__doc__ = """

Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

from .datasets_xyz import ImageDataset_XYZ, ImageDataset_XYZ_unpaired
from .datasets_rgb import ImageDataset_sRGB, ImageDataset_sRGB_unpaired
from .dataset_hdr import ImageDataset_HDRplus, ImageDataset_HDRplus_unpaired
from .dataset_xintu import ImageDatasetXinTu, ImageDatasetXinTuUnpaired


def build_dataset(cfg, model):
    dataset = DATASET_ARCH_REGISTRY.get(cfg.DATALOADER.DATASET)(cfg, model)
    return dataset
