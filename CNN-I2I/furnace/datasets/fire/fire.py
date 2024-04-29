#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : crowd.py
import numpy as np

from datasets.BaseDataset import BaseDataset


class Flame(BaseDataset):
    trans_labels = [0, 1]

    @classmethod
    def get_class_colors(*args):
        return [[0, 0, 0], [128, 0, 0]]

    @classmethod
    def get_class_names(*args):
        return ['Non-fire', 'Fire']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name