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

        self.patch_num = 288

        self.pos_embed = Parameter(torch.FloatTensor(1, self.patch_num, 2048))
        init.trunc_normal_(self.pos_embed.data, mean=0.0, std=0.02)

        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, g, cam_embed=None):
        q = x #self.dropout(x)
        rel_q = self.q(q)

        B, N, C = g.size()
        g = g.reshape(B * N // self.patch_num, self.patch_num, C)
        
        if cam_embed is not None:
            cam_embed = cam_embed.reshape(B * N // self.patch_num, 1, C)

        rel = F.normalize(g, dim=-1) # B x HW x C
        rel = torch.bmm(rel, rel.transpose(1, 2)) # B x HW x HW
        rel = torch.softmax(rel, dim=-1)

        g = torch.bmm(rel, g)
        rel_g = g + self.pos_embed
        if cam_embed is not None:
            rel_g = g + cam_embed
        g = g.reshape(B, N, C)
        rel_g = rel_g.reshape(B, N, C)

        rel_g = self.g(rel_g)

        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        # rel = torch.sigmoid(rel)
        rel = torch.softmax(rel, dim=-1)

        out = torch.bmm(rel, g)

        return out, rel
        

class deltaor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = 2048 ** -0.5

        self.normx = nn.LayerNorm(dim, elementwise_affine=False)
        self.normg = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)
        
        # self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, g):
        q = self.normx(x)
        rel_q = self.q(q)

        g = g - x.mean(dim=1, keepdim=True)#.clone().detach()
        g = self.normg(g)
        rel_g = self.g(g)

        rel = torch.bmm(rel_q, rel_g.transpose(1, 2)) # B x HW x HW
        rel = rel * self.scale
        rel = torch.softmax(rel, dim=-1)

        out = x + torch.bmm(rel, g)

        return out


class circle_fine_grained_extractor(nn.Module):
    def __init__(self, dim, q_num, rank):
        super().__init__()
        
        self.P2R = self.Q2R = catcher(dim)
        self.R2P = deltaor(dim)
        # catcher(dim)

        self.rank = rank
        self.num_instance = 4
        
        self.query_v = Parameter(torch.FloatTensor(1, q_num, dim))
        self.query_i = Parameter(torch.FloatTensor(1, q_num, dim))
        init.trunc_normal_(self.query_v.data, mean=0.0, std=0.02)
        self.query_i.data = self.query_v.data

        self.prototype = Parameter(torch.FloatTensor(1, 2048, dim))
        init.trunc_normal_(self.prototype.data, mean=0.0, std=0.02)

        self.weights = Parameter(torch.ones(1, q_num, 1))

        self.cam_embed = Parameter(torch.FloatTensor(6, 1, dim))
        init.trunc_normal_(self.cam_embed.data, mean=0.0, std=0.02)

    f_rec, f_cor, rel = None, None, None
    def forward(self, x, cam_ids, sub):
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, H*W).transpose(1, 2)

        # query = torch.cat([self.query_v.repeat(x.size(0), 1, 1)[sub], self.query_i.repeat(x.size(0), 1, 1)[~sub]], dim=0)
        query = self.query_v.repeat(x.size(0), 1, 1)
        prototype = self.prototype.repeat(x.size(0), 1, 1)  # B x N x C

        cam_embed = self.cam_embed[cam_ids-1]
        f_rel, _ = self.Q2R(query, x, cam_embed)

        f_pro = self.R2P(f_rel, prototype)
        
        f_rec, f_cor = None, None
        if self.training:
            f_rec, _ = self.P2R(f_pro, x, cam_embed)

            cam_embed = cam_embed.reshape(B // self.num_instance, 1, self.num_instance, 1, C)    # B/N x 1 x N x HW x C
            cam_embed = cam_embed.repeat(1, self.num_instance, 1, 1, 1)                       # B/N x N x N x HW x C
            cam_embed = cam_embed.reshape(B, self.num_instance, C)                        # B x NHW x C
            cam_embed = torch.cat([cam_embed[~sub], cam_embed[sub]], dim=0)
            
            x = x.reshape(B // self.num_instance, 1, self.num_instance, H*W, C)    # B/N x 1 x N x HW x C
            x = x.repeat(1, self.num_instance, 1, 1, 1)                       # B/N x N x N x HW x C
            x = x.reshape(B, self.num_instance*H*W, C)                        # B x NHW x C
            x = torch.cat([x[~sub], x[sub]], dim=0)

            f_cor, _ = self.P2R(f_pro, x, cam_embed)

        f_pro = torch.softmax(self.weights, dim=1) * f_pro

        return f_rel, f_pro, f_rec, f_cor
