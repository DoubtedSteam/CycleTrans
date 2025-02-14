import torch
from torch import nn
from torch.nn import functional as F
from utils.metric_dist import pairwise_distance


class CenterClusterLoss(nn.Module):
    def __init__(self, margin=0):
        super(CenterClusterLoss, self).__init__()
        self.margin = margin


    def forward(self, feats, labels, subs):
        # B = feats.size(0)
        # feats = feats.reshape(B, -1, 2048)
        # H = feats.size(1)

        # centers = []
        # for i in range(B):
        #     centers.append(feats[(labels == labels[i])].mean(0))
        # centers = torch.stack(centers)

        # dist_aps = (feats - centers).pow(2).sum(-1).clamp(min=1e-12).sqrt() # B x H

        # mask = ~(labels.expand(B, B).eq(labels.expand(B, B).t()))
        
        # loss = 0
        # for i in range(H):
        #     dist_ap = dist_aps[:, i]
        #     dist_ap = dist_ap.expand(B, B)
        #     dist_ap = dist_ap + dist_ap.t() # f_ij i和j到各自类心的距离之和

        #     dist_aa = pairwise_distance(feats[:, i, :], feats[:, i, :]) # 各个类心之间的距离

        #     loss += (dist_ap - dist_aa + self.margin)[mask].clamp(min=0.0).mean()

        # loss = loss / H
        
        B = feats.size(0)
        feats = feats.reshape(B, -1, 2048)

        centers = []
        for i in range(B):
            centers.append(feats[(labels == labels[i])].mean(0))
        centers = torch.stack(centers)

        dist_ap = (feats - centers).pow(2).sum(-1).clamp(min=1e-12).sqrt().sum(-1)
        dist_ap = dist_ap.expand(B, B)
        dist_ap = dist_ap + dist_ap.t() # f_ij i和j到各自类心的距离之和

        dist_aa = pairwise_distance(feats, feats) # 各个类心之间的距离

        mask = ~(labels.expand(B, B).eq(labels.expand(B, B).t()))

        loss = (dist_ap - dist_aa + self.margin)[mask].clamp(min=0.0).mean()
        
        return loss