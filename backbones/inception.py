import torchvision.models as models
import torch.nn as nn

def inception_v3(pretrained:bool=True, input_channels:int=1):
    model = models.inception_v3(pretrained=pretrained)
    if input_channels != 3:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(input_channels,32,
            kernel_size=(3,3),stride=(2,2), bias=False)

    model = nn.Sequential(
        *list(model.children())[:-3]
    )
    model.input_size = 299
    model.out_features = 2048
    model.name = 'inception_v3'

    return model

if __name__ == "__main__":
    model = inception_v3()
    print(model)