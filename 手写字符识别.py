# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:41:42 2022

@author: 20601
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # 导入数字库
from sklearn.model_selection import train_test_split  # 分割数据集
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier,plot_tree
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier#集成学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA#主成分分析方法

my_digit = load_digits()

x = my_digit.data
y = my_digit.target


acc_ada = []
acc_bag = []
n=range(2,50,2)
for i in n:
    X=PCA(n_components=10).fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf_ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i),n_estimators=100)
    clf_bag=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=i),n_estimators=100)
    clf_ada.fit(x_train,y_train)
    clf_bag.fit(x_train,y_train)
    acc_ada.append(clf_ada.score(x_test,y_test))
    acc_bag.append(clf_bag.score(x_test,y_test))
    
plt.plot(n,acc_ada,'r-o',n,acc_bag,'b-o')
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend('AdaBoost','Bagging')
    
    


#贝叶斯方法
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# clf_gnb = GaussianNB()  # 高斯朴素贝叶斯
# clf_mnb = MultinomialNB()  # 多项式朴素贝叶斯
# clf_bnb = BernoulliNB()  # 伯努利朴素贝叶斯

# n = range(1, 100, 10)

# acc1 = []
# acc2 = []
# acc3 = []

# for i in n:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=i)
#     clf_gnb.fit(x_train, y_train)
#     clf_mnb.fit(x_train, y_train)
#     clf_bnb.fit(x_train, y_train)
#     acc1.append(clf_gnb.score(x_test, y_test))
#     acc2.append(clf_mnb.score(x_test, y_test))
#     acc3.append(clf_bnb.score(x_test, y_test))

# plt.plot(n, acc1, 'bo--', label='GaussianNB')
# plt.plot(n, acc2, 'yo-', label='MultinomialNB')
# plt.plot(n, acc3, 'ro-', label='BernoulliNB')
# plt.xlabel('random_state')
# plt.ylabel('accuracy')
# # plt.legend(['GaussianNB','MultinomialNB','BernoulliNB'])
# plt.legend()
# plt.show()

# acc1=[]
# acc2=[]
# for i in range(1,2):
#     tree1=DecisionTreeClassifier(criterion='gini')
#     tree2=DecisionTreeClassifier(criterion='entropy')
#     tree1.fit(x_train,y_train)
#     tree2.fit(x_train,y_train)
#     acc1.append(tree1.score(x_test,y_test))
#     acc2.append(tree2.score(x_test,y_test))
# plt.figure(dpi=1000,figsize=(20,8))
# a=plot_tree(tree1,max_depth=(3),fontsize=(6))
# plt.savefig(args, kwargs)

# plt.plot(range(1,20),acc1,'b--',range(1,20),acc2,'r--')
# plt.xlabel('depth')
# plt.ylabel('accuracy')
# plt.title('tree')
# plt.grid(True)

# clf=LogisticRegression()
# clf.fit(x_train,y_train)
# clf.predict(x_test)

# print(clf.score(x_test,y_test))

# for i in len(x_test):

# clf=MLPClassifier(hidden_layer_sizes=(10,),activation="logistic")
# clf.fit(x_train,y_train)

# mlp1= {"hidden_layer_sizes": [(10,),(100,),(100,30),(100,30,10)],
#        "activation":["logistic","relu",'identity','tanh'],
#                              "solver": ['adam', 'sgd', 'lbfgs'],
#                              "max_iter": [200,10],
#                              "verbose": [False],
#                              "random_state":[0,1,2,3,4,5,6,7,8,9,10]
#                              }
# grid_search=GridSearchCV(MLPClassifier(random_state=0),param_grid=mplp1)
# grid_search.fit(x_train,y_train)
# print(grid_search.score(x_test,y_test))
# print grid_search.mlp1.keys()

# y_predict=clf.predict(x_test)
# print(confusion_matrix(y_test, y_predict))
# print(clf.score(x_test,y_test))
