# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:16:04 2022

@author: 20601
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical  # 对类别进行one—hot编码
from sklearn.model_selection import train_test_split

# 导入数据
data = []
target = []
# 正常数据
path1 = r"samples\normal"
data_1 = os.listdir(path1)
for a in range(len(data_1) - 1):
    data.append(plt.imread(path1 + "//" + data_1[a]))
    target.append(1)
# 块状数据
path2 = r"samples\block"
data_2 = os.listdir(path2)
for b in range(len(data_2) - 1):
    data.append(plt.imread(path2 + "//" + data_2[b]))
    target.append(2)
# 经向数据
path3 = r"samples\warp"
data_3 = os.listdir(path3)
for c in range(len(data_3) - 1):
    data.append(plt.imread(path3 + "//" + data_3[c]))
    target.append(3)
# 纬向数据
path4 = r"samples\weft"
data_4 = os.listdir(path4)
for d in range(len(data_4) - 1):
    data.append(plt.imread(path4 + "//" + data_4[d]))
    target.append(4)
data = np.array(data)  # 训练数据
target = np.array(target)  # 标签
data.reshape(899, 128, 256, 1)
target = to_categorical(target, 10)

# 设置数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

# conv2 = Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
# pool = MaxPooling2D()(conv2)
# flat = Flatten()(pool)
# dense = Dense(20, activation='relu')(flat)
# out = Dense(10, activation='softmax')(dense)
# my_model = Model(inputs=input, outputs=out)
# my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# my_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
# print(my_model.evaluate(x_test, y_test))
# plot_model(my_model, "model.png")

input = Input(shape=[128, 256, 1])
conv = Conv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(input)  # 共享卷积层
conv1 = Conv2D(filters=64, kernel_size=(7, 7), padding='same', activation='relu')(conv)
pool_1 = MaxPooling2D()(conv1)
flat1 = Flatten()(pool_1)

conv2 = Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='tanh')(conv)
pool_2 = MaxPooling2D()(conv2)
flat2 = Flatten()(pool_2)

final = concatenate([flat1, flat2])
dense1 = Dense(20, activation='relu')(final)
dense2 = Dense(30, activation='relu')(dense1)
out = Dense(10, activation='softmax')(dense2)

my_model = Model(inputs=input, outputs=out)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
my_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
print(my_model.evaluate(x_test, y_test))
plot_model(my_model, "model.png")
