import cv2
import torch
import numpy as np
from torch.utils import data

from config import config
from utils.img_utils import random_scale, random_scale_i2i, random_mirror_i2i, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

class TrainPre_Crowd(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt, i2i_gt):
        self.im_size = img.shape

        if (self.im_size[1] >= config.image_width) or (self.im_size[0] >= config.image_height):
            img = cv2.resize(img, (config.image_width, config.image_height), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)
            i2i_gt = cv2.resize(i2i_gt, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)

            

            img, gt, i2i_gt = random_mirror_i2i(img, gt, i2i_gt)
            img = normalize(img, self.img_mean, self.img_std)
            i2i_gt = normalize(i2i_gt, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))

        else:
            img, gt, i2i_gt = random_mirror_i2i(img, gt, i2i_gt)
            if config.train_scale_array is not None:
                img, gt, i2i_gt, scale = random_scale_i2i(img, gt, i2i_gt, config.train_scale_array)

            img = normalize(img, self.img_mean, self.img_std)
            i2i_gt = normalize(i2i_gt, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))

            w1, w2, h1, h2 = 0, 0, 0, 0

            if self.im_size[1] - config.image_width < 0:
                w1 = (config.image_width - self.im_size[1]) // 2
                w2 = (config.image_width - self.im_size[1]) - w1

            if self.im_size[0] - config.image_height < 0:
                h1 = (config.image_height - self.im_size[0]) // 2
                h2 = (config.image_height - self.im_size[0]) - h1

            img = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=0)
            gt = cv2.copyMakeBorder(gt, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=255)
            i2i_gt = cv2.copyMakeBorder(i2i_gt, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=[0.9, 0.9, 0.9])

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_i2i_gt, _ = random_crop_pad_to_shape(i2i_gt, crop_pos, crop_size, [0.9, 0.9, 0.9])
        p_i2i_gt = cv2.resize(p_i2i_gt, (config.image_width // 8,
                                         config.image_height // 8),
                              interpolation=cv2.INTER_NEAREST)

        p_img = p_img.transpose(2, 0, 1)
        p_i2i_gt = p_i2i_gt.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, p_i2i_gt, extra_dict

def get_train_loader_Crowd(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre_Crowd(config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)


    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=False,
                                   sampler=train_sampler)

    return train_loader, train_sampler

class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_gt = cv2.resize(p_gt, (config.image_width // config.gt_down_sampling,
                                 config.image_height // config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
