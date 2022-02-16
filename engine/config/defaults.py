from fvcore.common.config import CfgNode

_C = CfgNode(new_allowed=True)

_C.MODEL = CfgNode(new_allowed=True)
_C.MODEL.WEIGHTS = ""
_C.MODEL.TRAINER = CfgNode(new_allowed=True)
_C.MODEL.DEVICE = 'cuda'


_C.SOLVER = CfgNode(new_allowed=True)
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.STEPS = (60000, 80000)
_C.SOLVER.MAX_ITER = 90000
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.REFERENCE_WORLD_SIZE = 0
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY
_C.SOLVER.CLIP_GRADIENTS = CfgNode({"ENABLED": False})
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


# ---------------------------------------------------------------------------- #
# data loader config
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CfgNode(new_allowed=True)
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode(new_allowed=True)
_C.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
_C.INPUT.MAX_SIZE_TRAIN = 1333

_C.INPUT.FORMAT = "BGR"


_C.OUTPUT_DIR = ''
_C.OUTPUT_LOG_NAME = __name__


def get_defaults():
    return _C.clone()
