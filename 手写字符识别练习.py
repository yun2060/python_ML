# -*- coding: utf-8 -*-
"""
Created on Sun May 29 08:21:59 2022

@author: 20601
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets#导入手写字符识别数据
data=datasets.load_digits()

from sklearn.linear_model import LogisticRegression#逻辑回归
clf_lr=LogisticRegression()
from sklearn.tree import DecisionTreeClassifier#决策树
clf_tree=DecisionTreeClassifier()
from sklearn.svm import SVC#支持向量机
clf_svm=SVC()
from sklearn.neural_network import MLPClassifier#人工神经网络
clf_nn=MLPClassifier(hidden_layer_sizes=(666,),activation='logistic')

x=data.data
y=data.target
accurate1=[];accurate2=[];accurate3=[];accurate4=[];
for i in range(100,1500,100):
    trainx=x[:i,:]#numpy类型行和列用逗号分隔
    trainy=y[:i]
    testx=x[i:,:]
    testy=y[i:]
    clf_lr.fit(trainx,trainy)
    clf_tree.fit(trainx,trainy)
    clf_svm.fit(trainx,trainy)
    clf_nn.fit(trainx,trainy)
    accurate1.append(clf_tree.score(testx,testy))
    accurate2.append(clf_svm.score(testx,testy))
    accurate3.append(clf_nn.score(testx,testy))
    accurate4.append(clf_lr.score(testx,testy))
    # predicted1=clf_tree.predict(testx)
    # predicted2=clf_svm.predict(testx)
    # predicted3=clf_nn.predict(testx)
plt.plot(range(100,1500,100),accurate1,'ro-',label='tree')
plt.plot(range(100,1500,100),accurate2,'bo-',label='svm')
plt.plot(range(100,1500,100),accurate3,'yo-',label='nn')
plt.plot(range(100,1500,100),accurate4,'yo--',label='lr')
plt.title('MLmodel')
plt.xlabel('number of training')
plt.yticks(np.arange(0.45,1,0.05))
plt.ylabel('accuracy')
plt.legend()
plt.savefig('MLmodel.png',dpi=1000)

# import numpy as np
# from tensorflow.keras.datasets import mnist#导入mnist库
# from tensorflow.keras.models import Sequential#使用Keras中的Sequential模型（一层一层顺序执行）
# from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten
# #层：二维卷积、二维最大池化、全连接层、一维化（flatten用于卷积层和全连接层之间，扁平化）
# from tensorflow.python.keras.utils import np_utils#用于后续one_hot编码
# from sklearn.model_selection import train_test_split#用于分开训练集和测试集

# x=mnist.load_data()[0][0].reshape(-1,28,28,1)#训练数据
# y1=mnist.load_data()[0][1]#标签
# y=np_utils.to_categorical(num_classes=10,y=y1)#进行one_hot编码
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)#将test和train分开
# x_train=x_train.reshape(36000,28,28,1)[:1000]#取出训练集前1000张图片用作训练
# x_test=x_test.reshape(24000,28,28,1)[:500]#取出测试集前500张图片用作测试
# x_train=np.array(x_train)
# y_train=np.array(y_train)[:1000]#取出与训练集对应的标签
# x_test=np.array(x_test)
# y_test=np.array(y_test)[:500]#取出与测试集对应的标签


# model=Sequential()#定义model
# model.add(Conv2D(10,kernel_size=(3,3),padding='valid',activation='relu',input_shape=[28,28,1]))
#kernel_size是卷积核的大小，strides是卷积核滑动的步长,padding是边缘填充方法，input_shape指输入的张量
# model.add(MaxPool2D())#可加可不加，不使用池化的话，对步长进行调参
# model.add(Flatten())
# model.add(Dense(20,activation='relu'))
# model.add(Dense(10,activation='softmax'))#输出层激活函数只能是softmax
# model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=20,verbose=2,batch_size=64)
# print(model.evaluate(x_test,y_test))



# from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt
# x=mnist.load_data()[0][0]
# y=mnist.load_data()[0][1]






















