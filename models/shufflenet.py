import torchvision.models as models
import torch.nn as nn

def shufflenet(pretrained:bool=True, num_of_classes:int=30, input_channels:int=1):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 24, kernel_size=(3,3), stride=2, padding=1, bias=False),
        *model.conv1[1:]
    )
    model.fc = nn.Linear(1024,num_of_classes)
    model.input_size = 224
    return model

if __name__ == "__main__":
    print(shufflenet())