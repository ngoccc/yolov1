# Configurations & Hyper-parameters
# import torch
# from pathlib import Path
from easydict import EasyDict as edict

args = edict()

# basic options 
args.name = 'main'                   # experiment name.
args.resume = False                  # whether to resume. If you want to resume training, change this option.
args.ckpt_dir = 'ckpts'              # checkpoint directory name.
args.ckpt_reload = '10'              # If you want to resume training, specify which epoch's checkpoint to re-load.
args.gpu = True                      # whether or not to use gpu. 

# network options
# args.num_filters = 32                # number of output channels in the first nn.Conv2d module in MyNetwork.
# args.resblock_type = 'plain'         # type of residual block. ('plain' | 'bottleneck').
# args.num_resblocks = [1, 2, 3]       # number of residual blocks in each Residual Layer.
# args.use_bn = True                   # whether or not to use batch normalization.

# data options
# args.dataroot = ''                   # where data exist.
args.num_classes = 20
args.train_size = [416, 416]         # training img size
args.val_size = [416, 416]         # training img size
args.batch_size = 32                 # number of mini-batch size.
args.num_workers = 8

# training options
args.lr = 0.001                      # learning rate.
args.epoch = 100                     # training epoch.
args.momentum = 0.9
args.weight_decay = 5e-4

# tensorboard options
args.tensorboard = True              # whether or not to use tensorboard logging.
args.log_dir = 'logs'                # to which tensorboard logs will be saved.
args.log_iter = 100                  # how frequently logs are saved