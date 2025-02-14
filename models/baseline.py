import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from .embednet import embed_net

import cv2
import itertools

from models.resnet import resnet50
# from models.vit import vit_base_patch16_224
from utils.calc_acc import calc_acc
from utils.rerank import pairwise_distance

from models.modules import *
from layers import TripletLoss
from layers import AngleAlignmentLoss
from layers import CenterClusterLoss
from layers import MMD_loss

from copy import deepcopy


class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=True, multiscale=False, finegrained=False, world_size=1, rank=0, part_num=8, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride

        self.backbone = embed_net()

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.finegrained = finegrained

        self.world_size = world_size
        self.rank = rank

        self.fine_feat_num = 0
        self.final_dim = 2048
        self.num_classes = num_classes
        self.num_head = 7

        self.finegrained = True
        if self.finegrained:
            self.MMF = circle_fine_grained_extractor(2048, self.num_head, rank)
            self.final_dim = 2048 * self.num_head

        self.bn_neck = nn.BatchNorm1d(self.final_dim)
        init.normal_(self.bn_neck.weight, 1.0, 0.01)
        init.zeros_(self.bn_neck.bias)

        self.classification = True
        if self.classification:
            self.classifier = nn.Linear(self.final_dim, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, 0, 0.001)
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.MMDloss = MMD_loss(kernel_num=5)

        if self.triplet:
            self.pro_cc_loss = CenterClusterLoss(margin=0.7)


    def agg_from_gpus(self, x):
        y = [torch.zeros_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(y, x)
        y[self.rank] = x
        y = torch.cat(y, 0)
        return y


    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids', None)
        subs = ((cam_ids == 3) + (cam_ids == 6))
                
        if self.training:
            inputs = torch.cat([inputs[subs], inputs[~subs]], 0)
            labels = torch.cat([labels[subs], labels[~subs]], 0)
            cam_ids = torch.cat([cam_ids[subs], cam_ids[~subs]], 0)
            subs = torch.cat([subs[subs], subs[~subs]], 0)
            feats = self.backbone(x1 = inputs[subs], x2 = inputs[~subs])
        else:
            if subs.sum() > 0:
                feats = self.backbone(x1 = inputs[subs], x2 = inputs[~subs], modal=1)
            else:
                feats = self.backbone(x1 = inputs[subs], x2 = inputs[~subs], modal=2)

        # if self.training:
        #     inputs = torch.cat([self.thermal_attn(feats[subs]), 
        #                         self.visible_attn(feats[~subs])], 0)
        # else:
        #     if subs.sum() > 0:
        #         feats = self.thermal_attn(feats)
        #     else:
        #         feats = self.visible_attn(feats)

        if self.finegrained:
            # modality relevant feature, prototype feature, reconstruct feature, cross sample feature
            f_rel, f_pro, f_rec, f_cor = self.MMF(feats, cam_ids, subs)

        if not self.training:
            feats = f_pro.reshape(f_pro.shape[0], -1)
            bn_feats = feats.clone()
            bn_feats = self.bn_neck(bn_feats.reshape(bn_feats.shape[0], -1))
                
            return torch.cat([feats.reshape(feats.shape[0], -1), bn_feats.reshape(bn_feats.shape[0], -1)], dim=1)

        ############### Tranin ###############

        loss = 0
        metric = {}
        feats = self.agg_from_gpus(feats)
        labels = self.agg_from_gpus(labels)
        subs = subs.long()
        subs = self.agg_from_gpus(subs)
        subs = (subs == 1)

        B = feats.size(0)

        if self.finegrained:
            f_rec = self.agg_from_gpus(f_rec).reshape(B, self.num_head, 2048)
            f_rel = self.agg_from_gpus(f_rel).reshape(B, self.num_head, 2048)
            f_cor = self.agg_from_gpus(f_cor).reshape(B, self.num_head, 2048)

            rec_loss = (f_rec - f_rel.clone().detach()).abs().mean()            
            loss += rec_loss * 1.0 
            metric.update({'cir-rec': rec_loss.data})
            
            mod_loss = (f_cor.clone().detach() - f_rel).pow(2).sum(-1).clamp(min=1e-12).sqrt().mean()
            loss += mod_loss * 0.15 
            metric.update({'cir-mod': mod_loss.data})

        feats = f_pro.reshape(B, -1)

        if self.finegrained:
            cc_loss = self.pro_cc_loss(feats, labels, subs)
            loss += cc_loss * 0.2 
            metric.update({'id-ccp': cc_loss.data})
            
        feats = self.bn_neck(feats)

        if self.finegrained:
            mr_feats = feats.reshape(B, self.num_head, 2048)[:, 1:, :]
            mr_feats = F.normalize(mr_feats, p=2, dim=-1)
            relation_matrix = torch.bmm(mr_feats, mr_feats.permute(0, 2, 1)).clamp(min=0.0)
            sep_loss = (torch.triu(relation_matrix, diagonal=1)).sum(1).sum(1).mean() / ((self.num_head - 2) * (self.num_head - 1) / 2)
            loss += sep_loss * 0.3 
            metric.update({'id-sep': sep_loss.data})
            
        if self.classification:
            logits = self.classifier(feats.reshape(B, -1))
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'id-ce': cls_loss.data})

            source = feats[subs]
            target = feats[~subs]
            MMD_loss = self.MMDloss(source, target)
            loss += MMD_loss * 0.7 
            
            metric.update({'MMD': MMD_loss.data}) 

        return loss, metric
