# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:01 2015

@author: X
模块：裁切目标及背景
      调整正负样本
输入：多幅经过配准的包含水坝的卫星图像，一幅人工标注的水坝位置图
输出：按照裁切窗口中目标占有比例分类存放的样本集
"""

import os
from PIL import Image, ImageEnhance
import numpy as np
import math


# 随机调节亮度来增加正样本
def Light(image_pil, times=10, dark=0.9, bright=1.1):
    # image_pil: Image格式的图像
    # times: 增加的倍数
    # dark: 最暗
    # bright：最亮
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


# 随机旋转来增大一个样本
def Rotate(image_pil, times=10, neg_angle=-2.0, pos_angle=2.0):
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


def TargetBox(image_np):
	# Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


# 自动采样
def AutoCrop(label_src,  # 标记图像目录
             image_src,  # 遥感影像目录
             dst_dir,  # 数据集保存目录
             window_size=(128, 128),  # 采样窗口尺寸
             stride=(32, 32),  # 采样步长
             radius=1,  # 采样窗口中心区域的半径
             max_pos=2000,  # 正样本指标
             neg_pos_rate=6.0):  # 负样本指标是正样本的倍数
    # 目标范围(行最小，列最小，行最大，列最大，行重心，列重心)
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass

if __name__ == "__main__":

    image_src = "raw_data/satellite"
    label_src = "raw_data/label"
    dst_dir = "Experiment/crop/dataset"
    window_size = (256, 256)
    stride = (32, 32)  # 1/8 window_size
    radius = 1
    dst_path = dst_dir
    AutoCrop(label_src, image_src, dst_path,
             window_size=window_size,
             stride=stride,
             radius=radius,
             max_pos=1000,
             neg_pos_rate=2.5)
