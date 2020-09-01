import torchvision.models as models
import torch.nn as nn

def resnet50(pretrained:bool=True, input_channels:int=1):
    model = models.resnet50(pretrained=pretrained)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels,64,
            kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    model = nn.Sequential(
        *list(model.children())[:-2]
    )
    model.input_size = 224
    model.out_features = 2048
    model.name = 'resnet50'
    return model

if __name__ == "__main__":
    model = resnet50()
    print(model)