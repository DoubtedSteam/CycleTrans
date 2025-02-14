import torch
from torch import nn
import torch.nn.functional as F

class AngleAlignmentLoss(nn.Module):
    def __init__(self):
        super(AngleAlignmentLoss, self).__init__()

    def forward(self, inputs, targets, subs):
        B = inputs.size(0)

        centers = []
        for i in range(B):
            centers.append(inputs[(targets == targets[i]) * (subs == subs[i])].mean(0))
        centers = torch.stack(centers) # 每一类每个模态的类心

        mask0 = ~(targets.expand(B, B).eq(targets.expand(B, B).t())) * (subs.expand(B, B).eq(0)) # 到同类A模态所有点
        mask1 = ~(targets.expand(B, B).eq(targets.expand(B, B).t())) * (subs.expand(B, B).eq(1)) # 到同类B模态所有点

        loss = 0
        for i in range(B):
            vec = inputs - centers[i:i+1, :] 
            vec = F.normalize(vec, p=2, dim=-1)
            ang = torch.matmul(vec, vec.t())
            loss += torch.abs(ang[mask0[i]] - ang[mask1[i]]).mean()
        loss /= B

        return loss
