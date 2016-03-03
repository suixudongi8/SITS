# -*- coding: utf-8 -*-
"""
Created on Fri Jan 01 18:26:16 2016

@author: X
控制整个算法流程
"""

import datetime
import time
from crop2 import AutoCrop
from train import Load_dataset, Train
from detect import DetectFull
now = datetime.datetime.now


def Control(codex="",
            max_pos=400,  # 正样本指标
            neg_pos_rate=4.0):  # 负样本指标是正样本的多少倍):

    t0 = now()

    date_time = time.strftime("%Y.%m.%d-%H%M", time.localtime())
    # Ex_code：实验代码，一个以日期-时间命名的文件夹
    Ex_code = "Experiment/" + codex + '/' + date_time

    print("#1. Crop and make dataset.")
    label_src = "raw_data/label"
    image_src = "raw_data/(0, 0)"
    data_to = Ex_code + "/dataset"
    window_size = (128, 128)
    stride = (8, 8)  # 1/8 window_size
    radius = 4  # 采样框的中心区域半径
    max_pos = max_pos  # 正样本指标
    neg_pos_rate = neg_pos_rate  # 负样本指标是正样本的多少倍

    success, tedge = AutoCrop(label_src, image_src, data_to,
                              window_size=window_size,
                              stride=stride,
                              radius=radius,
                              max_pos=max_pos,
                              neg_pos_rate=neg_pos_rate)
    if not success:
        print("Failed at Crop!")
        return False

    t1 = now()

    print("#2. Training.")
    data_from = data_to
    data_to = Ex_code+"/model"
    classifier_size = (128, 128)
    success, X, y = Load_dataset(data_from, classifier_size=classifier_size)
    if not success:
        print("Failed at Load dataset!")
        return False

    nb_filter = 8, 16, 24, 32
    kernel_size = 7, 5, 3, 3
    conv = [nb_filter, kernel_size]
    dense = 512,
    config = {'conv': conv, 'dense': dense}
    model_code = Train(X, y, data_to, config=config)

    t2 = now()

    print("#3. Detect.")
    test_dir = "test/(0, 0)"
    label_dir = Ex_code+"/A"
    resolution = 0.5
    DetectFull(test_dir, model_code, label_dir,
               resolution=resolution,
               target_edge=tedge,
               window_size=window_size,
               stride=stride,
               classifier_size=classifier_size,
               iter_times=2,
               n_random_error=2,
               radius=32)

    t3 = now()

    # 记录时间
    fname = str("%s_t1-%f_t2-%f_t3-%f" % (Ex_code,
                                          (t1-t0).total_seconds(),
                                          (t2-t1).total_seconds(),
                                          (t3-t2).total_seconds()))
    with open(fname, 'wb'):
        pass

    return True


if __name__ == "__main__":
    for t in range(1):
#        for max_pos in [800, 1200]:
        for max_pos in [2400]:
            for rate in [10]:
                codex = str("%d x %d 2d" % (max_pos, rate))
                Control(codex=codex, max_pos=max_pos, neg_pos_rate=rate)
