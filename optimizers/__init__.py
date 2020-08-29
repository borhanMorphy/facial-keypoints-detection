from torch import optim
from typing import Dict

__optimizer_mapper__ = {
    'ADAM':{
        'cls': optim.Adam,
        'kwargs': {
            'lr':1e-3,
        }
    },

    'SGD':{
        'cls': optim.SGD,
        'kwargs': {
            'lr': 1e-3,
            'momentum': 0.9 
        }
    }
}

def get_optimizer(optimizer_name:str, parameters, kwargs:Dict={}) -> optim.Optimizer:
    assert optimizer_name in __optimizer_mapper__,"given optimizer is not valid"

    optim_cls = __optimizer_mapper__[optimizer_name]['cls']
    optim_configs = __optimizer_mapper__[optimizer_name]['kwargs']
    # re-write
    optim_configs.update(kwargs)
    return optim_cls(parameters, **optim_configs)