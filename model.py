import torch
import torch.nn as nn
import torch.nn.functional as F

class FacialKeypointsDetector(nn.Module):
    def __init__(self, backbone:nn.Module, criterion=None,
            device:str='cpu', num_classes:int=30):

        super(FacialKeypointsDetector,self).__init__()

        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(backbone.out_features, num_classes)
        self.criterion = criterion
        self.device = device
        self.name = f"fkd_{backbone.name}"
        self.to(device)

    def forward(self, data:torch.Tensor):
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        features = self.backbone(data) # N, out_features, x, x
        features = self.pool(features).flatten(start_dim=1) # N, out_features
        preds = self.head(features) # N, num_classes
        return torch.sigmoid(preds) # activate with sigmoid to fit between [0,1]

    def training_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is not training than switch to training mode
        if not self.training: self.train()
        preds = self.forward(data.to(self.device))
        loss = self.criterion(preds, targets.to(self.device))
        return loss

    def val_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is training than switch to eval mode
        if self.training: self.eval()
        with torch.no_grad():
            preds = self.forward(data.to(self.device))
            loss = self.criterion(preds, targets.to(self.device))
        return loss

    def test_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is training than switch to eval mode
        if self.training: self.eval()
        with torch.no_grad():
            # TODO do not use criterion, use competition metric
            preds = self.forward(data.to(self.device))
            loss = self.criterion(preds, targets.to(self.device))
        return loss

    def get_input_size(self):
        return self.backbone.input_size
