import torch


def get_model_state_dict(model):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
