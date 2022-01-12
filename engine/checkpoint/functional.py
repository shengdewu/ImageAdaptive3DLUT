import torch
import logging
import copy
from engine import TORCH_VERSION


def _match_model_state_dict(model, checkpoint, log_name=None):

    checkpoint_state_dict = copy.deepcopy(checkpoint)
    model_state_dict = get_model_state_dict(model)

    incorrect_shapes = []
    missing_keys = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, torch.nn.parameter.UninitializedParameter
            ):
                continue
            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            missing_keys.append(k)

    if log_name is not None:
        if missing_keys:
            logging.getLogger(log_name).warning('the model missing_keys keys \n {}'.format(missing_keys))
        if incorrect_shapes:
            logging.getLogger(log_name).warning('the model incorrect_shapes keys \n {}'.format(incorrect_shapes))

    return checkpoint_state_dict


def get_model_state_dict(model):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict, log_name=None):
    match_state_dict = _match_model_state_dict(model, state_dict, log_name)
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        incompatible = model.module.load_state_dict(match_state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(match_state_dict, strict=False)

    if log_name is not None:
        if incompatible.missing_keys:
            logging.getLogger(log_name).warning('the model missing keys \n {}'.format(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logging.getLogger(log_name).warning('the model unexpected keys \n {}'.format(incompatible.unexpected_keys))


def load_checkpoint_state_dict(model, state_dict):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
