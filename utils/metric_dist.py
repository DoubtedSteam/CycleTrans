import numpy as np
import torch


def pairwise_distance(x, y): # x,y \in [B, C]
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def softmax_pairwise_distance(x, y): # x,y \in [B, N, C]
    m, n = x.size(0), y.size(0)
    assert x.size(1) == y.size(1)
    p = x.size(1)
    x = x.transpose(0, 1) # N x B x C
    y = y.transpose(0, 1)
    dist = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(p, m, n) + \
            torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(p, n, m).transpose(1, 2)
    dist = dist - 2 * torch.bmm(x, y.transpose(1, 2)) # N x B x C
    dist = dist.clamp(min=1e-12).sqrt()
    weight = torch.softmax(dist, dim=0)
    dist = (dist * weight).sum(0)
    return dist


def anti_softmax_pairwise_distance(x, y): # x,y \in [B, N, C]
    m, n = x.size(0), y.size(0)
    assert x.size(1) == y.size(1)
    p = x.size(1)
    x = x.transpose(0, 1) # N x B x C
    y = y.transpose(0, 1)
    dist = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(p, m, n) + \
            torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(p, n, m).transpose(1, 2)
    dist = dist - 2 * torch.bmm(x, y.transpose(1, 2)) # N x B x C
    dist = dist.clamp(min=1e-12).sqrt()
    weight = 1 - torch.softmax(dist, dim=0)
    dist = (dist * weight).sum(0)
    return dist


def smooth_pairwise_distance(x, y): # x,y \in [B, N, C]
    m, n = x.size(0), y.size(0)
    x = x.reshape(m, -1, 2048)
    y = y.reshape(n, -1, 2048)
    p = x.size(1)
    x = x.transpose(0, 1) # N x B x C
    y = y.transpose(0, 1)
    dist = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(p, m, n) + \
            torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(p, n, m).transpose(1, 2)
    dist = dist - 2 * torch.bmm(x, y.transpose(1, 2)) # N x B x C
    dist = dist.clamp(min=1e-12).sqrt().sum(0)
    return dist