import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import init
from torch.nn import Parameter

import math
import random


class catcher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 2048 ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)

    def forward(self, x, g):
        q = x + g.mean(dim=1, keepdim=True)
        rel_q = self.q(q)

        B, N, C = g.shape
        g = g.reshape(B * N // 192, 192, C)

        rel = F.normalize(g, dim=-1) # B x HW x C
        rel = torch.bmm(rel, rel.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        rel = torch.softmax(rel, dim=-1)

        g = torch.bmm(rel, g)
        g = g.reshape(B, N, C)
        rel_g = self.g(g)

        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        rel = torch.softmax(rel, dim=-1)

        out = torch.bmm(rel, g)

        return out
        

class deltaor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 2048 ** -0.5

        self.normx = nn.LayerNorm(dim)
        self.normg = nn.LayerNorm(dim)
        # self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)

    def forward(self, x, g):
        q = self.normx(x)
        rel_q = self.q(q)

        g = g - x.mean(dim=1, keepdim=True)#.clone().detach()
        g = self.normg(g)
        rel_g = self.g(g)

        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        rel = torch.softmax(rel, dim=-1)

        out = q + torch.bmm(rel, g)

        # out = self.norm(out)

        return out


class circle_fine_grained_extractor(nn.Module):
    def __init__(self, dim, q_num, hw, num_heads, rank):
        super().__init__()
        
        self.Q2R = catcher(dim)
        self.R2P_v = deltaor(dim)
        self.R2P_i = deltaor(dim)
        self.P2R = catcher(dim)

        self.rank = rank
        self.num_instance = 4

        self.query_v = Parameter(torch.FloatTensor(1, q_num, dim)) # 1 x N x C
        self.query_i = Parameter(torch.FloatTensor(1, q_num, dim))
        init.trunc_normal_(self.query_v.data, std=0.02)
        self.query_i.data = self.query_v.data

        self.prototype = Parameter(torch.FloatTensor(1, 1024, 2048))
        init.trunc_normal_(self.prototype.data, mean=0.0, std=0.8)

        # self.drop = nn.Dropout()

    def forward(self, x, sub):
        B, C, H, W = x.size()
        
        x = x.reshape(B, C, H*W).transpose(1, 2)

        query = torch.cat([self.query_v.repeat(x.size(0), 1, 1)[sub], self.query_i.repeat(x.size(0), 1, 1)[~sub]], dim=0)          # B x M x C
        prototype = self.prototype.repeat(x.size(0), 1, 1)  # B x N x C
        # prototype = self.drop(prototype)

        f_rel = self.Q2R(query, x)
        f_rel = torch.cat([x.mean(dim=1, keepdim=True), f_rel], dim=1)

        if (f_rel[sub].size(0) > 0):
            f_pro_v = f_pro = self.R2P_v(f_rel[sub], prototype[sub])
        if (f_rel[~sub].size(0) > 0):
            f_pro_i = f_pro = self.R2P_i(f_rel[~sub], prototype[~sub])
        if (f_rel[sub].size(0) > 0) and (f_rel[~sub].size(0) > 0):
            f_pro = torch.cat([f_pro_v, f_pro_i], dim=0)

        # f_pro = self.R2P(f_rel, prototype)
        
        if self.training:
            x_mate = x.reshape(B // self.num_instance, 1, self.num_instance, H*W, C)    # B/N x 1 x N x HW x C
            x_mate = x_mate.repeat(1, self.num_instance, 1, 1, 1)                       # B/N x N x N x HW x C
            x_mate = x_mate.reshape(B, self.num_instance*H*W, C)                        # B x NHW x C
            x_mate = torch.cat([x_mate[~sub], x_mate[sub]], dim=0)

            f_cor = self.P2R(f_pro, x_mate)

            index = list(range(B // 2))
            index = torch.LongTensor(index).to(x.device)
            index = index + self.num_instance
            index = torch.fmod(index, B // 2)

            x_v = x[sub]
            x_r = x[~sub]
            x_v = torch.cat([x_v, x[sub][index]], dim=1)
            x_r = torch.cat([x_r, x[~sub][index]], dim=1)
            x = torch.cat([x_v, x_r], dim=0)

            f_rec = self.P2R(f_pro, x)

            return f_rel, f_pro, f_rec, f_cor

        else:
            return f_rel, f_pro, None, None
