import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class WeightedPairLoss(nn.Module):
    def __init__(self, margin=1.):
        super(WeightedPairLoss, self).__init__()
        # self.weights = Parameter(torch.FloatTensor(dim_num, class_num))
        self.margin = 0.5

    def forward(self, inputs, targets, subs):
        B, C = inputs.shape # B x C
        inputs_ = inputs.expand(B, B, C).clone().detach()
        diff = torch.abs(inputs_ - inputs_.permute(1, 0, 2))
        mean_diff = diff.mean(-1, keepdim=True).repeat(1, 1, C)

        identity_mask = targets.expand(B, B).eq(targets.expand(B, B).t())
        modality_mask = subs.expand(B, B).eq(subs.expand(B, B).t())
        loss = 0

        # Find out the dimensions about modality (Same ID Same Modality Small Difference)
        modality_diff = diff[identity_mask * modality_mask].reshape(B, -1, C)
        mean_diff = modality_diff.mean(-1, keepdim=True).repeat(1, 1, C)
        modality_diff_mask = (modality_diff < mean_diff * self.margin).sum(1)
        modality_diff_mask = modality_diff_mask.reshape(B, 1, C).repeat(1, B, 1)
        modality_diff_mask = (modality_diff_mask * modality_diff_mask.permute(1, 0, 2)).clone().detach().float()

        # Find out the dimensions about identity (Same ID Same Modality Large Difference)
        identity_diff = diff#[~identity_mask].reshape(B, -1, C)
        mean_diff = identity_diff.mean(-1, keepdim=True).repeat(1, 1, C)
        identity_diff_mask = (identity_diff < mean_diff * self.margin)#.sum(1) # B x C
        # identity_diff_mask = identity_diff_mask.reshape(B, 1, C).repeat(1, B, 1)
        # identity_diff_mask = (identity_diff_mask * identity_diff_mask.permute(1, 0, 2)).clone().detach().float()

        for i in range(B):
            diff = inputs[i:i+1] - inputs
            
            modality_diff = diff * modality_diff_mask[i]
            modality_diff = modality_diff.pow(2).sum(-1).clamp(min=1e-12).sqrt().max()

            identity_diff = diff * identity_diff_mask[i]
            identity_diff = identity_diff[~identity_mask[i]]
            identity_diff = identity_diff.pow(2).sum(-1).clamp(min=1e-12).sqrt().min()

            loss += (modality_diff * 10 - identity_diff).clamp(min=0.0)

        loss /= B

        return loss