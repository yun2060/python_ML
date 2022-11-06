# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:35:00 2022

@author: 20601
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=78)
# 天气：晴天1，阴天2，小雨3
# 温度：高1，中2，低3
# 湿度：高1，中2，低3
# 刮风：是1，否2
# 坤：是1，否2
x = np.array(([1, 1, 2, 2], [1, 1, 2, 1], [2, 1, 1, 2], [3, 1, 1, 2], [3, 3, 1, 2], [1, 2, 2, 1], [1, 2, 1, 1]))
y = ([2], [2], [1], [1], [2], [1], [2])
accurate = []
for n in range(1, 5):
    trainx = x[:n, :]  # numpy类型行和列用逗号分隔
    trainy = y[:n]
    testx = x[n:, :]
    testy = y[n:]

    clf.fit(trainx, trainy)
    accurate.append(clf.score(testx, testy))

plt.plot(range(1, 5), accurate)
plt.show()