import torch
import math
from torch import nn
from torch.nn import functional as F

def pairwise_distance(f1, f2):
    n = f1.size(0)
    m = f2.size(0)
    dist = torch.pow(f1, 2).sum(dim=1, keepdim=True).expand(n, m) \
         + torch.pow(f2, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist.addmm_(1, -2, f1, f2.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.3, modality=1.0, scale=1.0, num_classes=395, k_size=4, total=140):
        super(AdaptiveTripletLoss, self).__init__()
        self.margin = margin
        self.learning_modality = modality
        self.learning_scale = scale

        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.k_size = k_size
        self.scale = 0
        self.total_epoch = total

    def forward(self, inputs, targets, subs, epoch):
        n = inputs.size(0)
        
        # Limit the search area
        ids_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        sub_mask = subs.expand(n,n).eq(subs.expand(n,n).t())
        
        mask1 =  ids_mask * ~sub_mask   # same identity         different modality
        mask2 = ~ids_mask * ~sub_mask   # different identity    different modality
        mask3 = ~ids_mask *  sub_mask   # different identity    same modality

        dist = pairwise_distance(inputs, inputs)
        current_scale = dist[sub_mask].max()
        if self.scale == 0:
            self.scale = current_scale

        # Find the closest cross modality pair for each identity
        if self.learning_modality > 0.0:
            n_pair = []
            m = self.k_size // 2
            for i in range(0, n, self.k_size):
                d, index = dist[targets == targets[i]][mask1[targets == targets[i]]].view(-1, 1).min(dim=0)
                index = index[0].item()
                u = i + index // m
                v = (index % m) * 2 + i
                if u % 2 == 0:
                    v += 1
                if subs[u]:
                    n_pair.append((u, v))
                else:
                    n_pair.append((v, u))
        n_pair = torch.LongTensor(n_pair)

        loss = 0
        real_ap, real_an, prec, direc = [], [], [], []
        for i in range(n):
            
            # Learning identity
            g_ap, i_pos = dist[i][ids_mask[i]].max(dim=0)
            
            # Consider direction
            # direction = inputs - inputs[i]
            # direction = direction / (direction * direction).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
            # direction = 1 + (1 - epoch / self.total_epoch) * (direction * (direction[ids_mask[i]])[i_pos]).sum(dim=1)
            # g_an, i_neg = (direction[~ids_mask[i]]).min(dim=0)
            # direc.append(g_an)
            # g_an = (dist[i][~ids_mask[i]])[i_neg]

            # Without direction
            g_an, _ = dist[i][~ids_mask[i]].min(dim=0)
            direc.append(g_an)

            real_ap.append(g_ap)
            real_an.append(g_an)

            identity_learning = (self.margin + g_ap - g_an).clamp(min=0.0)

            # Learning modality
            modality_learning = 0
            if self.learning_modality > 0:
                tmp_mask = ~((n_pair[:, 0].eq(i)) + (n_pair[:, 1].eq(i)))
                m_pair = n_pair[tmp_mask, :]

                if ~sub_mask[i][m_pair[0][0]]:
                    modality_learning += (dist[i][m_pair[:, 0]] - dist[i][m_pair[:, 1]]).clamp(min=0.0)
                else:
                    modality_learning += (dist[i][m_pair[:, 1]] - dist[i][m_pair[:, 0]]).clamp(min=0.0)
                modality_learning = modality_learning.sum()

            # Summary
            loss += identity_learning + self.learning_modality * modality_learning

        real_ap = torch.stack(real_ap)
        real_an = torch.stack(real_an)
        direc = torch.stack(direc)
        
        loss = loss / n
        current_scale = current_scale.item()
        self.scale = self.scale * 0.95 + current_scale * 0.05
        
        return loss#, real_ap.mean(), real_an.mean(), direc.mean().item()