# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:03:24 2022

@author: 20601
"""
from sklearn import datasets#导入手写字符识别数据
data=datasets.load_digits()
import numpy as np
import matplotlib.pyplot as plt
# plt.imshow(data.data[89].reshape((8,8)),'gray')
# data.target[89]

from sklearn.tree import DecisionTreeClassifier#导入决策树（方法一）
from sklearn import svm

#from sklearn import tree#导入树库（方法二）
# # 创建树对象
# clf = tree.DecisionTreeClassifier(criterion='gini')  
# #clf = tree.DecisionTreeRegressor() for regression#使用决策树拟合

x=data.data   #numpy类型
y=data.target

n=1400  #存放训练样本数量

trainx=x[:n,:]#numpy类型行和列用逗号分隔
trainy=y[:n]
testx=x[n:,:]
testy=y[n:]

acc1=[]
acc2=[]
acc3=[]

for c in np.linspace(0.1,1,10):
    model = svm.SVC(C=c)
    model.fit(trainx,trainy)
    acc3.append(model.score(testx,testy))

for i in range(1,11):
# # 利用训练集对模型进行训练，并检查评分
    clf1=DecisionTreeClassifier(criterion='gini',max_depth=(i),)# 对于分类，这里可以将算法更改为基尼或熵(信息增益)，默认为基尼
    clf1.fit(trainx,trainy)
    acc1.append(clf1.score(testx,testy))

    clf2=DecisionTreeClassifier(criterion='entropy',max_depth=(i))# 对于分类，这里可以将算法更改为基尼或熵(信息增益)，默认为基尼
    clf2.fit(trainx,trainy)
    acc2.append(clf2.score(testx,testy))
# #输出预测
# predicted=clf.predict(testx)

plt.plot(range(1,11),acc1,range(1,11),acc2,range(1,11),acc3)
plt.xlabel('depth of decisiontree')
plt.ylabel('accuracy')
plt.title('this is an example')
plt.legend(['gini','entropy','svm'])
plt.grid(True)
