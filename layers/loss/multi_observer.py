from builtins import print
import torch
import math
from torch import nn
from torch.nn import functional as F

class MultiObserverLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MultiObserverLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, subs):
        B, C = inputs.size()
        
        centers = []
        for i in range(B // 4):
            centers.append(inputs[(targets == targets[i * 4])].mean(0))
        centers = torch.stack(centers)

        identity_mask = targets.expand(B, B).eq(targets.expand(B, B).t()) # same identity
        modality_mask = subs.expand(B, B).eq(subs.expand(B, B).t()) # same modality

        losses = 0
        for i in range(B // 4):
            vec = inputs - centers[i:i+1]

            vec = F.normalize(vec, p=2, dim=-1)

            angle = torch.matmul(vec, vec.t())

            inter_postive = angle[identity_mask].reshape(B, -1).min(dim=-1, keepdim=True)[0] # B x 1
            inter_postive = torch.exp(-inter_postive)

            # print(inter_postive.shape)
            # print(inter_postive.max())
            # print(inter_postive.min())

            intra_negtive = angle[~identity_mask].reshape(B, -1).clamp(min=0.0).max(dim=-1, keepdim=True)[0]
            intra_negtive = torch.exp(intra_negtive)

            # print(intra_negtive.shape)
            # print(intra_negtive.max())
            # print(intra_negtive.min())

            # loss = (inter_postive * intra_negtive).clamp(min=0.0).mean()
            loss = torch.matmul(inter_postive.t(), intra_negtive).mean() / (8 * 56)

            losses += loss

        losses = losses / B * 4
        
        return loss