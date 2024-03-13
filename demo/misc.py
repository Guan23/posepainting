#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/14 下午3:43
import time
import os

import numpy as np

from load_data import *
# from project import main
import myconfig as mcfg


# 统计所有训练集中，pedestrian、cyclist和person_sitting三种类别的样本数量
def gt_object_num(maxnum=1779):
    frame_pedestrian = 0
    num_pedestrian = 0
    easy_pedestrian = 0
    moderate_pedestrian = 0
    hard_pedestrian = 0
    other = 0
    filenames = os.listdir(os.path.join(mcfg.ROOT_SPLIT_PATH, "image_2"))
    filenames.sort()
    for i, idx in enumerate(filenames):
        if i >= maxnum:
            break
        have_pedestrian = False
        idx = idx.split(".")[0]
        # print("第{}帧数据，编号为:{}".format(i + 1, idx))
        gt_label_lists = get_label(idx)
        for gt_label in gt_label_lists:
            if gt_label.cls_id == 1:
                have_pedestrian = True
                if gt_label.level == 0:
                    easy_pedestrian += 1
                elif gt_label.level == 1:
                    moderate_pedestrian += 1
                elif gt_label.level == 2:
                    hard_pedestrian += 1
                else:
                    other += 1
        if have_pedestrian:
            frame_pedestrian += 1
    num_pedestrian = easy_pedestrian + moderate_pedestrian + hard_pedestrian
    return frame_pedestrian, num_pedestrian, easy_pedestrian, moderate_pedestrian, hard_pedestrian, other


# num_pedestrian: 4487
# frame_pedestrian: 1779
# num_cyclist: 736
# num_person_sitting: 205
# total: 5428
#
# easy_pedestrian: 2325
# moderate_pedestrian: 1258
# hard_pedestrian: 708
# 4291

if __name__ == "__main__":
    print("\n------------------------ start ------------------------\n")
    print(gt_object_num(maxnum=1779))
    print("\n------------------------- end -------------------------\n")
