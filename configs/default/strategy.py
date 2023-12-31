from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "baseline"
strategy_cfg.port = 12355

# setting for circle construction
strategy_cfg.prototype_num = 768

# setting for loader
strategy_cfg.sample_method = "random"
strategy_cfg.batch_size = 128
strategy_cfg.p_size = 16
strategy_cfg.k_size = 8

# setting for loss
strategy_cfg.classification = True
strategy_cfg.triplet = False
strategy_cfg.ap = False
strategy_cfg.center = False
strategy_cfg.adaptive_triplet = False
strategy_cfg.MMD = False
strategy_cfg.MMDmargin = False
strategy_cfg.MMDFC = False
# setting for metric learning
strategy_cfg.margin = 0.3
strategy_cfg.p = 0.0
# settings for optimizer
strategy_cfg.optimizer = "sgd"
strategy_cfg.stage = 'finetune'
strategy_cfg.lr = 0.1
strategy_cfg.wd = 5e-4
strategy_cfg.lr_step = [40]
strategy_cfg.warmup = 10
strategy_cfg.fp16 = False

strategy_cfg.num_epoch = 60

# settings for dataset
strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (384, 128)

# settings for augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = False
strategy_cfg.mixup = False
strategy_cfg.padding = 10

# settings for base architecture
strategy_cfg.drop_last_stride = False
strategy_cfg.label_smooth = False
strategy_cfg.finegrained = False
strategy_cfg.rerank = False

# logging
strategy_cfg.eval_interval = -1
strategy_cfg.start_eval = 60
strategy_cfg.log_period = 10

# testing
strategy_cfg.resume = ''