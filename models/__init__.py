from .shufflenet import shufflenet

__model_mapper__ = {
    'shufflenet': shufflenet
}

def get_model(model_name:str, pretrained:bool=True, num_of_classes:int=30):
    assert model_name in __model_mapper__

    return __model_mapper__[model_name](
        pretrained=pretrained,
        num_of_classes=num_of_classes)