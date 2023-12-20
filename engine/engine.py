import torch
import numpy as np

# from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast

def create_train_engine(model, optimizer, rank, worldsize, scaler, non_blocking=False, p_size=8, k_size=8):
    device = torch.device("cuda:"+str(rank))

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        b = data.size(0) // worldsize
        data = data[b * rank: b * (rank+1)]
        labels = labels[b * rank: b * (rank+1)]
        cam_ids = cam_ids[b * rank: b * (rank+1)]

        epoch = engine.state.epoch

        iteration = engine.state.iteration

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()

        with autocast():
            loss, metric = model(data, labels,
                                cam_ids=cam_ids,
                                epoch=epoch,
                                iteration=iteration)
        
        # loss.backward()
        scaler.scale(loss).backward()

        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        
        if rank != 0:
            metric = {}

        return metric

    return Engine(_process_func)


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def create_eval_engine(model, rank=0, non_blocking=False):
    device = torch.device("cuda:"+str(rank))

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths, img_ids = batch

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            # feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))
            feat = model.module(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking)) \
                 + model.module(fliplr(data), cam_ids=cam_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
