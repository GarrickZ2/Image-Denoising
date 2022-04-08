import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, x, y):
        loss = F.l1_loss(x, y, reduction='mean')        
        return loss * 1000
      

class Smooth_L1_Loss(nn.Module):
    def __init__(self):
        super(Smooth_L1_Loss, self).__init__()

    def forward(self, x, y):
        loss = F.smooth_l1_loss(x, y, reduction='mean')        
        return loss * 1000


class L1_L2_Loss(nn.Module):
    def __init__(self, ratio):
        super(L1_L2_Loss, self).__init__()
        self.ratio = ratio

    def forward(self, x, y):
        L1_loss = F.l1_loss(x, y, reduction='mean')
        L2_loss = F.mse_loss(x, y, reduction='mean')
        L1_L2 = (self.ratio)*L1_loss + (1-self.ratio)*L2_loss
        
        return L1_L2
