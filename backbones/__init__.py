from .shufflenet import shufflenet

__backbone_mapper__ = {
    'shufflenet': shufflenet
}

def get_backbone(backbone_name:str, pretrained:bool=True):
    assert backbone_name in __backbone_mapper__

    return __backbone_mapper__[backbone_name](pretrained=pretrained)


def get_available_backbones():
    return list(__backbone_mapper__.keys())