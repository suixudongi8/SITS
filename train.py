# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:53:43 2015

@author: X
训练分类器
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from PIL import Image
import numpy as np
from keras.utils import np_utils
from six.moves import range

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

import json
now = datetime.datetime.now


# 采样
def Sampling(src_dir, classifier_size=(64, 64), label=0):
    if not os.path.exists(src_dir):
        print("srcPath doesn't exists!")
        return [], []

    samples = os.listdir(src_dir)
    samples_len = len(samples)
    src_path = src_dir + '/'

    samples_list, labels_list = [], []
    for nth in range(samples_len):
        print("%d/%d" % (nth+1, samples_len))
        # 获取基本参数
        sample_name = src_path + samples[nth]
        image_pil = Image.open(sample_name)
        # 适配分类器输入尺寸
        small = image_pil.resize(classifier_size, resample=Image.ANTIALIAS)
        image_np = np.asarray(small, dtype=np.uint8)
        # Image风格：img_rows, img_cols, img_channels
        # Keras CNN风格：img_channels, img_rows, img_cols
#        sample = [image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]]
        sample = image_np.transpose(2, 0, 1)
        samples_list.append(sample)
        labels_list.append(label)

    return samples_list, labels_list


# 把训练数据从硬盘加载到内存中
def Load_dataset(dataset_dir, classifier_size=(64, 64), n_class=2):
    if not os.path.exists(dataset_dir):
        print("No dataset_dir!!!")
        return False, [], []
    print("Load dataset...")
    # 采集样本
    samples = []
    labels = []
    for nc in range(n_class):
        pos_samples, pos_labels = Sampling(dataset_dir+str("/%d" % (nc)),
                                           classifier_size=classifier_size,
                                           label=nc)
        if len(pos_samples) == 0:
            print("Failed when sampling class: %d" % nc)
            return False, [], []
        samples += pos_samples
        labels += pos_labels
    samples = np.asarray(samples)
    labels = np.asarray(labels)

    # 打乱数据
    print("Shuffle data...")
    index = [i for i in range(len(labels))]
    np.random.seed()
    np.random.shuffle(index)
    np.savetxt(dataset_dir+"/index.txt", index, fmt="%d")

    samples = samples[index]
    labels = labels[index]
    labels = np_utils.to_categorical(labels, n_class)

    samples = samples.astype("float32", copy=False)
    samples /= 255

    return True, samples, labels


# 使用DCNN进行训练
def Train(X, y, save_model_to, batch_size=128, split=0.2, config=[]):

    img_channels, img_rows, img_cols = X.shape[1:]
    model_str = ""
    conv = config["conv"]
    dense = config["dense"]
    nb_filter = conv[0]
    kernel_size = conv[1]
    n_class = y.shape[-1]

    # 构建卷积神经网络
    print("Modeling CNN...")
    model = Sequential()

    # 添加卷积层
    n_conv = len(nb_filter)
    for nl in range(n_conv):
        if nl == 0:
            model.add(Convolution2D(nb_filter[nl],
                                    kernel_size[nl], kernel_size[nl],
                                    border_mode='same',
                                    input_shape=(img_channels,
                                                 img_rows, img_cols)))
            model_str += str("C%dx%d-" % (nb_filter[nl], kernel_size[nl]))
            model.add(Activation('relu'))

#            model.add(Convolution2D(nb_filter[0],
#                                    kernel_size[0], kernel_size[0]))
#            model_str += str("C%2dx%d-" % (nb_filter[0], kernel_size[0]))
#            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
#            model.add(Dropout(0.25))
        else:
            model.add(Convolution2D(nb_filter[nl],
                                    kernel_size[nl], kernel_size[nl],
                                    border_mode='same'))
            model_str += str("C%dx%d-" % (nb_filter[nl], kernel_size[nl]))
            model.add(Activation('relu'))

#            model.add(Convolution2D(nb_filter[nl],
#                                    kernel_size[nl], kernel_size[nl]))
#            model_str += str("C%2dx%d-" % (nb_filter[nl], kernel_size[nl]))
#            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
#            model.add(Dropout(0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())

    for output_dim in dense:
        model.add(Dense(output_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model_str += str("F%d-" % output_dim)

    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    model_str += str("c%d" % n_class)

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    print("Comliling CNN...")
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model_path = save_model_to + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_code = model_path + model_str
    model_file = model_code + ".json"
#    weight_file = model_code + "_{epoch:02d}-{val_loss:.7f}-{val_acc:.7f}.hdf5"
    weight_file = model_code + ".hdf5"
    # 加入适时停止回掉
    check = ModelCheckpoint(filepath=weight_file, save_best_only=True)
    estop = EarlyStopping(monitor='val_loss', patience=2)

    # 训练
    print("Training...")
    # 此处可以加入预训练的权重
#    model.load_weights("M_2015.12.13_0233_0.0264696111471_0.99027313194.h5")

    t = now()
    model.fit(X, y,
              validation_split=split,
              batch_size=batch_size,
              callbacks=[check, estop],
              verbose=1,
              show_accuracy=True)
    print('Training time: %s' % (now() - t).total_seconds())

    print("Saving model...")
    model_json = model.to_json()
    with open(model_file, 'wb') as mj:
        json.dump(model_json, mj)

    return model_code


if __name__ == "__main__":

    dataset_dir = "Experiment/400/2016.01.11-1717/dataset"
    classifier_size=(128, 128)
    success, X, y = Load_dataset(dataset_dir, classifier_size=classifier_size)
    if not success:
        pass
    else:

#    nb_filter = 4, 8, 12, 16, 384
#    kernel_size = 7, 5, 3, 3
#    config = [nb_filter, kernel_size]
#    model_dir = "Experiment/01.06-2000/model/4_4_384d0.25"
#    for tms in range(5):
#        model = Train(X, y, model_dir, config=config)

#    nb_filter = 8, 12, 16, 20, 512
#    kernel_size = 7, 5, 3, 3
#    config = [nb_filter, kernel_size]
#    model_dir = "Experiment/01.06-2000/model/8_4_512d0.25"
#    for tms in range(5):
#        model = Train(X, y, model_dir, config=config)

#    for dns in [256, 384, 512, 768, 1024]:
#        nb_filter = 8, 16, 24, 32, dns
#        kernel_size = 7, 5, 3, 3
#        config = [nb_filter, kernel_size]
#        model_dir = "Experiment/01.06-2000/model/C8_8_"+str(dns)
#        for tms in range(10):
#            model = Train(X, y, model_dir, config=config)
#    # 测试双隐层
#    for dns in [384]:
#        nb_filter = 8, 16, 24, 32, dns, dns
#        kernel_size = 7, 5, 3, 3
#        config = [nb_filter, kernel_size]
#        model_dir = "Experiment/01.06-2000/model/Dense"+str(dns)
#        for tms in range(10):
#            model = Train(X, y, model_dir, config=config)
#     测试Maxout
#    for dns in [384]:
#        nb_filter = 8, 16, 24, 32, dns, dns
#        kernel_size = 7, 5, 3, 3
#        config = [nb_filter, kernel_size]
#        model_dir = "Experiment/2016.01.11-2123/model/"+str(dns)
#        for tms in range(10):
#            model = Train(X, y, model_dir, config=config)

        nb_filter = 8, 16, 24, 32
        kernel_size = 7, 5, 3, 3
    #    layers = [(8, 7),
    #              (16, 5),
    #              (24, 3),
    #              (32, 3)]
        conv = [nb_filter, kernel_size]
        dense = 512,
        config = {'conv': conv, 'dense': dense}
        model_dir = "Experiment/2016.01.11-1624/model/"+str(dense)
        model_code = Train(X, y, model_dir, config=config)
