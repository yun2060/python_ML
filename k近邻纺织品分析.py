# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:16:04 2022

@author: 20601
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
data1 = []
data2 = []
data3 = []
data4 = []
im1 = plt.imread(r"samples\block\1.bmp")
# # 正常数据
# for i in range(1, len(os.listdir(r"samples\normal")) + 1):
#     data1.append(plt.imread(r"samples\normal//" + str(i) + ".bmp"))
# data1 = np.array(data1)
# # 块状数据
# for j in range(1, len(os.listdir(r"samples\block")) + 1):
#     data2.append(plt.imread(r"samples\block//" + str(j) + ".bmp"))
# data2 = np.array(data2)
# # 经向数据
# for k in range(1, len(os.listdir(r"samples\warp")) + 1):
#     data3.append(plt.imread(r"samples\warp//" + str(k) + ".bmp"))
# data3 = np.array(data3)
# # 纬向数据
# for m in range(1, len(os.listdir(r"samples\weft")) + 1):
#     data4.append(plt.imread(r"samples\weft//" + str(m) + ".bmp"))
# data4 = np.array(data4)

path1 = r"samples\normal"
data_1 = os.listdir(path1)
for i in range(len(data_1) - 1):
    data1.append(plt.imread(path1 + "//" + data_1[i]))

# 设置数据集
x = data1[100]

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import plot_model
