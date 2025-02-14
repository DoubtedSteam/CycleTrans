import os
import logging
import torch
import numpy as np
from sklearn.preprocessing import normalize
from .rerank import re_ranking, pairwise_distance
from utils.metric_dist import smooth_pairwise_distance
from torch.nn import functional as F

def softmax_pairwise_distance(x, y): # x,y \in [B, N, C]
    m, n = x.size(0), y.size(0)
    assert x.size(1) == y.size(1)
    # x = x.view(m, -1, 2048)
    # y = y.view(n, -1, 2048)
    x = x.view(m, 1, -1)
    y = y.view(n, 1, -1)
    p = x.size(1)
    x = x.transpose(0, 1) # N x B x C
    y = y.transpose(0, 1)
    dist = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(p, m, n) + \
          torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(p, n, m).transpose(1, 2)

    dist = dist - 2 * torch.bmm(x, y.transpose(1, 2)) # N x B x C
    dist = dist.clamp(min=1e-12).sqrt()
    weight = torch.softmax(dist / 10, dim=0)
    dist = (dist * weight).sum(0)
    return dist

def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])

    return names


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=1, num_trials=10, rerank=False):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2
    query_feats_bbn = query_feats[:, :2048 * 7]
    query_feats_bn = F.normalize(query_feats[:, 2048 * 7:], dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)
    
    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats_bbn = gallery_feats[:, :2048 * 7]
    gallery_feats_bn = F.normalize(gallery_feats[:, 2048 * 7:], dim=1)
    
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feat_bbn = gallery_feats_bbn[flag]
        g_feat_bn = gallery_feats_bn[flag]
        
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        if rerank:
            dist_mat = re_ranking(query_feats, g_feat)
        else:
            dist_mat =  1.0 * pairwise_distance(query_feats_bn[:, :2048], g_feat_bn[:, :2048]) \
                       + 0.5 * sum(pairwise_distance(query_feats_bn[:, 2048*i: 2048*(i+1)], g_feat_bn[:, 2048*i: 2048*(i+1)]) for i in range(1, 7)) \
                       + 0.1 * pairwise_distance(query_feats_bbn[:, :2048], g_feat_bbn[:, :2048]) \
            # dist_mat = pairwise_distance(query_feats_bn, g_feat_bn) \
            #          + 0.5 * pairwise_distance(query_feats_bn[:, :2048], g_feat_bn[:, :2048])
                 
            
        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = '{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(mode, num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
