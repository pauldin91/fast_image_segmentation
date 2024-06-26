# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

import torch.utils.model_zoo as model_zoo

C = edict()
config = C
cfg = C

C.seed = 12345
C.gpu_id = '1' # change
C.gpu_number = 'cuda:'+C.gpu_id # change

C.repo_name = '~/fast_image_segmentation/CNN-I2I'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'furnace'))

from base_config.base_config import base_config

C.channels = base_config.channels
C.dataset = base_config.dataset
C.network = base_config.network
C.resnet = base_config.resnet
C.image_mean = base_config.image_mean
C.image_std = base_config.image_std
C.image_height = base_config.image_height
C.image_width = base_config.image_width
C.image_size = base_config.image_size

C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))
C.Imgs_dir = osp.abspath(osp.join(C.log_dir, 'Imgs')) # change

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = "~/D-Fire/"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "lists/train.txt")
C.eval_source = osp.join(C.dataset_path, "lists/test.txt")
C.test_source = osp.join(C.dataset_path, "lists/test.txt")
C.is_test = False

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'furnace'))

# from furnace.utils.pyt_utils import model_urls

"""Image Config"""
C.num_classes = 2
C.background = 0
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.target_size = 640
C.image_height = 480  # 768
C.image_width = 640  # 768 * 2
C.gt_down_sampling = 8
# C.gt_down_sampling = 16  # *2 for MobileNetV2
C.num_train_imgs = 1341
C.num_eval_imgs = 662

""" Settings for network, this would be different for each kind of model"""
C.fix_bias = True
C.fix_bn = False
C.sync_bn = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1
#C.pretrained_model = None
C.pretrained_model = "~/fast_image_segmentation-main/CNN-I2I/resnet18_v1.pth"

"""Train Config"""
C.lr = 1e-3
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 5e-4
C.batch_size = 4  # 4 * C.num_gpu
C.nepochs = 120
C.niters_per_epoch = 500
C.num_workers = 1
C.train_scale_array = [0.75, 1, 1.25, 1.5, 1.75, 2.0]
# C.train_scale_array = [1, 1.25, 1.5, 1.75, 2.0]
# C.train_scale_array = None

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 5 / 6
C.eval_scale_array = [1, ]
C.eval_flip = False
C.eval_height = 480
C.eval_width = 640

"""Display Config"""
C.snapshot_iter = 50
C.record_info_iter = 20
C.display_iter = 50

C.GPUS = (1,)
C.MODEL_DEEP = False


def open_tensorboard():
    pass


if __name__ == '__main__':
    print(config.epoch_num)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
