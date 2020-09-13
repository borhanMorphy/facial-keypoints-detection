import torch.nn as nn
from typing import Dict
from .rmse import RMSELoss

__criterion_mapper__ = {
    'MSE':{
        'cls': nn.MSELoss,
        'kwargs': {}
    },

    'RMSE':{
        'cls': RMSELoss,
        'kwargs':{}
    },

    'L1':{
        'cls': nn.L1Loss,
        'kwargs': {}
    },

    'SL1':{
        'cls': nn.SmoothL1Loss,
        'kwargs': {}
    }
}

def get_criterion(criterion_name:str, kwargs:Dict={}):
    assert criterion_name in __criterion_mapper__,"given criterion is not valid"

    loss_cls = __criterion_mapper__[criterion_name]['cls']
    loss_configs = __criterion_mapper__[criterion_name]['kwargs']
    # re-write
    loss_configs.update(kwargs)
    return loss_cls(**loss_configs)


def get_available_criterions():
    return list(__criterion_mapper__.keys())