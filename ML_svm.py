# -*- coding: utf-8 -*-
"""
Created on Sun May 29 08:19:22 2022

@author: 20601
"""

#Import Library
from sklearn import svm
# 创建SVM分类对象 

model = svm.SVC() # 有各种与它相关的选项，这是简单的分类。你可以参考链接，了解更多细节。
 
#利用训练集对模型进行训练，并检查评分
model.fit(X, y)
model.score(X, y)
 
# #输出预测
predicted= model.predict(x_test)