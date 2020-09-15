import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import get_criterion
from typing import Tuple

class DoubleStageRegressor(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super(DoubleStageRegressor,self).__init__()
        self.fc1 = nn.Linear(in_features,out_features)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(out_features,out_features)
        
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        x = self.fc1(x)
        s1 = self.act(x)
        s2 = self.fc2(x)
        return s1,s2

class FacialKeypointsDetector(nn.Module):
    def __init__(self, backbone:nn.Module, criterion=None,
            device:str='cpu', num_classes:int=30, alpha:float=.2):

        super(FacialKeypointsDetector,self).__init__()
        self.alpha = alpha
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = DoubleStageRegressor(backbone.out_features, num_classes)
        self.criterion = criterion
        self.rmse = get_criterion("RMSE")
        self.device = device
        self.name = f"fkd_{backbone.name}"
        self.to(device)

    def forward(self, data:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        features = self.backbone(data) # N, out_features, x, x
        features = self.pool(features).flatten(start_dim=1) # N, out_features
        s1,s2 = self.head(features) # (N, num_classes),(N, num_classes)
        
        return s1,s2

    def training_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is not training than switch to training mode
        if not self.training: self.train()
        s1,s2 = self.forward(data.to(self.device))
        s1_loss = self.criterion(s1, targets.to(self.device))
        s2_loss = self.criterion(s2, targets.to(self.device))
        return s1_loss + s2_loss*self.alpha

    def val_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is training than switch to eval mode
        if self.training: self.eval()
        with torch.no_grad():
            s1,s2 = self.forward(data.to(self.device))
            loss = self.rmse(s2, targets.to(self.device))
        return loss

    def test_step(self, data:torch.Tensor, targets:torch.Tensor):
        # if model mode is training than switch to eval mode
        if self.training: self.eval()
        with torch.no_grad():
            # TODO do not use criterion, use competition metric
            s1,s2 = self.forward(data.to(self.device))
            loss = self.criterion(s2, targets.to(self.device))
        return loss

    def predict(self, data:torch.Tensor):
        if self.training:self.eval()

        with torch.no_grad():
            _,s2 = self.forward(data.to(self.device))

        return s2


    def get_input_size(self):
        return self.backbone.input_size
