#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
from preprocess import load
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from symptom_train import symptom_predict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_and_test(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    # print("train_acc ", train_acc)
    # print("test_acc ", test_acc)
    return train_acc, test_acc


def xgboost_train(sym_pred=False):
    train_explicit, train_implicit, train_tag = load('train')
    dev_explicit, dev_implicit, dev_tag = load('dev')
    if sym_pred:
        train_explicit = train_explicit + train_implicit*symptom_predict(train_explicit)
        dev_explicit = dev_explicit + dev_implicit*symptom_predict(dev_explicit)
    xg_train = xgb.DMatrix(train_explicit, label=train_tag)
    xg_test = xgb.DMatrix(dev_explicit, label=dev_tag)
    # 1.训练模型
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 10
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 12

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 100
    bst = xgb.train(param, xg_train, num_round, watchlist)

    pred = bst.predict(xg_test)
    train_pred = bst.predict(xg_train)
    print('dev classification error=%f' % accuracy_score(dev_tag, pred))
    print('train classification error=%f' % accuracy_score(train_tag, train_pred))
    pickle.dump(bst, open("disease.pickle.dat", "wb"))


def disease_predict(X):
    xg_test = xgb.DMatrix(X)
    model = pickle.load(open("disease.pickle.dat", "rb"))
    return model.predict(xg_test)


# # LinearSVC
# print("LinearSVC")
# print("ovr")
# model = OneVsRestClassifier(svm.LinearSVC(random_state=0))
# print(train_and_test(model, train_explicit, train_tag, dev_explicit, dev_tag))
# '''
# train_acc  0.7269874476987448
# test_acc  0.6778242677824268
# '''
# model = OneVsOneClassifier(svm.LinearSVC(random_state=0))
# print(train_and_test(model, train_explicit, train_tag, dev_explicit, dev_tag))
# '''
# train_acc 0.747907949790795
# test_acc 0.6652719665271967
# '''
#
# # KNN
# print("KNN")
# for metric_type in ["minkowski", "manhattan", "chebyshev"]:
#     train_accs = []
#     test_accs = []
#     for k in range(1, 21):
#         model = KNeighborsClassifier(n_neighbors=k, metric=metric_type)
#         acc1, acc2 = train_and_test(model, train_explicit, train_tag, dev_explicit, dev_tag)
#         train_accs.append(acc1)
#         test_accs.append(acc2)
#     plt.plot(range(1, 21), train_accs, label='train_acc')
#     plt.plot(range(1, 21), test_accs, label='test_acc')
#     plt.title(metric_type)
#     plt.show()
# '''
# manhattan 5
# manhattan 0.7390167364016736
# manhattan 0.6861924686192469
# '''

# GBDT
# model = GradientBoostingClassifier()
# print(train_and_test(model, train_explicit+train_implicit, train_tag, dev_explicit, dev_tag))
'''
only explicit
(0.8368200836820083, 0.7447698744769874)
add implicit
(0.9712343096234309, 0.7071129707112971)
'''

# XGboost
'''
only explicit
train:0.8132
test:0.7573
add implicit
train:0.9806
test:0.874477
'''
if __name__ == '__main__':
    xgboost_train(True)
