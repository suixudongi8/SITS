# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 10:16:35 2015
使用大幅遥感图像开始实测道路检测
2015/12/4 9：38
加入求精过程
2015/12/29 14：37
修改入口
@author: X
"""

import os
from PIL import Image
import numpy as np
from keras.models import model_from_json
import json


dv = 0.989  # 属于某一类别的预测阈值
stroke_width = 6  # 标记矩形边框宽度
GPS_error_meter = 20.0

# 计算目标重心
def Center(image_np):
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


# 加载分类器
def Load_model(model_name):
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


# 根据坐标集和窗口大小采样
def SampleFromSet(image_pil, coordinates_list,
                  window_size=(256, 256),  # 采样窗口大小
                  classifier_size=(128, 128)):  # 分类器输入尺寸):
    # image_pil:一幅PIL Image格式的图像
    # coordinates_list:采样坐标集
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


# 检测一幅遥感影像
def DetectOne(image_pil,  # 待测遥感图像一幅
              classifier,  # 分类器
              resolution=1.0,  # 图像分辨率：米/像素
              target_edge=None,  # 目标区域范围
              window_size=(256, 256),  # 采样窗口大小
              stride=(32, 32),
              classifier_size=(128, 128),  # 分类器输入尺寸
              batch_size=256,   # 每一次送给分类器的样本数目
              iter_times=4,
              radius=1):  # 迭代求精次数
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass


def DetectFull(test_dir, model_name, label_dir,
               resolution=1.0,  # 图像分辨率：米/像素
               target_edge=None,  # 目标区域范围
               window_size=(256, 256),  # 采样窗口大小
               stride=(32, 32),
               classifier_size=(128, 128),  # 分类器输入尺寸
               batch_size=256,  # 每一次送给分类器的样本数目
               n_random_error=10,  # 随机产生几个有偏移的卫星影像
               iter_times=4,  # 迭代求精次数
               radius=1):
    # Please contact the author:
	# xudong2014@iscas.ac.cn
	# suixudongi8@qq.com
	pass

if __name__ == "__main__":
    test_dir = "test/Q"
    model = "Experiment/800 x 8 2d/2016.01.18-1557/model/C8x7-C16x5-C24x3-C32x3-F512-c2"  # 分类器名称
    label_dir = "Experiment/2016.01.06-2000/A"
    DetectFull(test_dir, model, label_dir,
               window_size=(256, 256),
               classifier_size=(128, 128),
               iter_times=4)
