#!/user/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from preprocess import load
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import *
from xgboost import XGBClassifier
import pickle


def train_and_test():
    train_explicit, train_implicit, train_tag = load('train')
    dev_explicit, dev_implicit, dev_tag = load('dev')
    train_symptom = np.array(train_implicit != 0, dtype=int)
    dev_symptom = np.array(dev_implicit != 0, dtype=int)

    clf_multilabel = MultiOutputClassifier(
        XGBClassifier(tree_method='gpu_hist', gpu_id=0, eval_metric='logloss', use_label_encoder=False))

    clf_multilabel.fit(train_explicit, train_symptom)
    val_pred = clf_multilabel.predict(dev_explicit)
    print("f1 score", f1_score(dev_symptom, val_pred, average='macro'))
    pickle.dump(clf_multilabel, open("symptom.pickle.dat", "wb"))


def symptom_predict(X):
    model = pickle.load(open("symptom.pickle.dat", "rb"))
    return model.predict(X)


if __name__ == '__main__':
    train_and_test()
