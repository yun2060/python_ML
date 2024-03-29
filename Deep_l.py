# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:03:20 2022

@author: 20601
"""
from tensorflow.keras.utils import plot_model
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential  # 序贯模型
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

x = mnist.load_data()[0][0][:1000]  # 训练数据
y = mnist.load_data()[0][1][:1000]  # 标签
hsy_x = x.reshape(1000, 28, 28, 1)
hsy_y = to_categorical(y, 10)
trainx = hsy_x[:800]
testx = hsy_x[800:]
trainy = hsy_y[:800]
testy = hsy_y[800:]

input1 = Input(shape=([28, 28, 1]))
conv11 = Conv2D(filters=20, kernel_size=(3, 3), padding='same', activation='relu')(input1)
pool11 = MaxPooling2D()(conv11)
conv12 = Conv2D(filters=10, kernel_size=(5, 5), padding='same', activation='relu')(pool11)
pool12 = MaxPooling2D()(conv12)
flat1 = Flatten()(pool12)

input2 = Input(shape=([28, 28, 1]))
conv21 = Conv2D(filters=15, kernel_size=(3, 3), padding='same', activation='relu')(input2)
pool21 = MaxPooling2D()(conv21)
conv22 = Conv2D(filters=5, kernel_size=(5, 5), padding='same', activation='relu')(pool21)
pool22 = MaxPooling2D()(conv22)
flat2 = Flatten()(pool22)

final = concatenate([flat1, flat2])
den1 = Dense(10, activation='relu')(final)
den2 = Dense(20, activation='relu')(den1)
output = Dense(10, activation='softmax',name='output')(den2)
my_model = Model(inputs=[input1, input2], outputs=output)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# my_model.fit([trainx, trainx], trainy, batch_size=32, epochs=10, verbose=2)
# print(my_model.evaluate([testx, testx], testy))
plot_model(my_model,'model.png')

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1)[:1000]
# x_test = x_test.reshape(-1, 28, 28, 1)[500:]
# y_train = to_categorical(y_train, 10)[:1000]
# y_test = to_categorical(y_test, 10)[500:]
# model = Sequential()
# model.add(Conv2D(filters=10, kernel_size=(3, 3), padding='same', input_shape=[28, 28, 1], activation='relu'))
# model.add(Conv2D(filters=15, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=15, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(20, activation='tanh'))
# model.add(Dense(15, activation='tanh'))
# model.add(Dense(10, activation='softmax'))
# # 各个损失函数是干啥的：mse\mape\msle.....各个优化器、
# model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
# print(model.evaluate(x_test, y_test))
# # plot_model(model,
# #                to_file='model.png',
# #                show_shapes=False,
# #                show_dtype=False,
# #                show_layer_names=True,
# #                rankdir='TB',
# #                expand_nested=False,
# #                dpi=96)
# plot_model(model, to_file='123.png', show_shapes=True)



# # 序贯模型
# my_cov = Sequential()
# my_cov.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=[28, 28, 1], activation='relu'))
# my_cov.add(MaxPooling2D())
# my_cov.add(Flatten())
# my_cov.add(Dense(20, activation='tanh'))
# my_cov.add(Dense(10, activation='softmax'))
# my_cov.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# my_cov.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
# print(my_cov.evaluate(x_test, y_test))

# 函数式模型
# my_input = Input(shape=[28, 28, 1])
# conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(my_input)  # 共享卷积层

# conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv)
# pool_1 = MaxPooling2D()(conv1)
# flat1 = Flatten()(pool_1)

# conv2 = Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='tanh')(conv)
# pool_2 = MaxPooling2D()(conv2)
# flat2 = Flatten()(pool_2)

# final = concatenate([flat1, flat2])
# dense1 = Dense(20, activation='relu')(final)
# dense2 = Dense(30, activation='relu')(dense1)
# out = Dense(10, activation='softmax')(dense2)

# my_model = Model(inputs=my_input, outputs=out)
# my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# my_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
# print(my_model.evaluate(x_test, y_test))
# plot_model(my_model, "model.png")

# x = mnist.load_data()[0][0][:2000]  # 训练数据
# y = mnist.load_data()[0][1][:2000]  # 标签
# x = x.reshape(2000, 28, 28, 1)
# x2 = x[:, ::2, ::2, :]
# x1_train = x[:1000]
# x1_test = x[1000:]
# y = to_categorical(y, 10)
# ytrain = y[:1000]
# ytest = y[1000:]
# x2_train = x2[:1000]
# x2_test = x2[1000:]



# import numpy as np
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
# from tensorflow.keras import Model
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils import np_utils
# from tensorflow.keras.utils import plot_model
