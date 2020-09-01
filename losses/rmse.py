import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self, reduction:str='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(input, target, reduction=self.reduction)
        return torch.sqrt(mse)