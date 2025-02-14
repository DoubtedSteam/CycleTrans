import logging
import os
import pprint

import torch
import yaml
from torch.cuda import amp
from torch import optim

from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from engine.scheduler import CosineLRScheduler, WarmUpMultiStepLR
from models.baseline import Baseline

from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
# from apex.parallel import DistributedDataParallel as DDP

def train(cfg, world_size, rank):
    # set logger
    logger = None
    writer = None
    if rank == 0:
        log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # writer = SummaryWriter(log_dir)
        writer = None

        logging.basicConfig(format="%(asctime)s %(message)s",
                            filename=log_dir + "/" + "log.txt",
                            filemode="w")

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        logger.info(pprint.pformat(cfg))

    # training data loader
    train_loader, sampler = get_train_loader(dataset=cfg.dataset,
                                    root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    world_size=world_size,
                                    rank=rank,
                                    num_workers=8)

    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0 and rank == 0:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)

    # model
    model = Baseline(num_classes=cfg.num_id,
                     drop_last_stride=cfg.drop_last_stride,
                     label_smooth=cfg.label_smooth,
                     triplet=cfg.triplet,
                     adaptive_triplet=cfg.adaptive_triplet,
                     margin=cfg.margin,
                     classification=cfg.classification,
                     finegrained=cfg.finegrained,
                     prototype_num=cfg.prototype_num,
                     world_size=world_size,
                     rank=rank,
                     k_size=cfg.k_size,
                     MMD=cfg.MMD,
                     MMDmargin=cfg.MMDmargin,
                     MMDFC=cfg.MMDFC,
                     ap=cfg.ap,
                     p=cfg.p
                     )

    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'init_model.pth'))
    torch.distributed.barrier()
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'init_model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # optimizer
    assert cfg.optimizer in ['adam', 'sgd']
    assert cfg.stage in ['pretrain', 'finetune']

    optimizer = optim.Adam([
        {'params':[param for name, param in model.named_parameters() if 'MMF' not in name], 'lr': 0.00035},
        {'params':[param for name, param in model.named_parameters() if 'MMF' in name],     'lr': 0.00035},
    ], weight_decay=cfg.wd)
        
    lr_scheduler = WarmUpMultiStepLR(optimizer=optimizer,
                                    milestones=cfg.lr_step,
                                    warmup_t=10,
                                    gamma=0.1)
    
    # base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    # optimizer = optim.SGD([
    #     {'params':[param for name, param in model.named_parameters() if 'classifier' not in name], 'lr': 0.01},
    #     {'params':[param for name, param in model.named_parameters() if 'classifier' in name],     'lr': 0.1},
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        
    # lr_scheduler = WarmUpMultiStepLR(optimizer=optimizer,
    #                                 milestones=[20, 80, 120],
    #                                 warmup_t=10,
    #                                 gamma=0.1)

    # engine
    scaler = GradScaler()
    engine = get_trainer(dataset=cfg.dataset,
                         model=model,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         writer=writer,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         start_eval=cfg.start_eval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader,
                         rank=rank,
                         worldsize=world_size,
                         sampler=sampler,
                         scaler=scaler,
                         rerank=cfg.rerank,
                         p_size=cfg.p_size,
                         k_size=cfg.k_size)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)

def start(rank, world_size, cfg):
    print('Running DDP on rank %d'%(rank))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.port)
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    train(cfg, world_size, rank)

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    args = parser.parse_args()

    # set random seed
    seed = 1
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # enable cudnn backend
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()

    world_size = torch.cuda.device_count()
    mp.spawn(
        start,
        args=(world_size, cfg, ),
        nprocs=world_size,
        join=True
    )

    # train(cfg, 1, 0)
