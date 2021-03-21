#!/user/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from preprocess import load
from symptom_train import symptom_predict
from disease_train import disease_predict
from preprocess import get_slot_set, get_disease_set
from copy import deepcopy
import pickle

test_explicit, test_implicit = load('test')
pred_symptom = symptom_predict(test_explicit)
test_explicit = test_explicit + pred_symptom * test_implicit
pred_disease = disease_predict(test_explicit)
slot_set = np.array(get_slot_set())
disease_set = np.array(get_disease_set())
submit_ans = []
for symptoms, disease in zip(pred_symptom, pred_disease):
    one_dic = {"symptom": [], "disease": ""}
    symptoms = list(slot_set[np.nonzero(symptoms)])
    one_dic["symptom"] = symptoms
    one_dic["disease"] = disease_set[int(disease)]
    submit_ans.append(deepcopy(one_dic))
    print(one_dic)
print(len(submit_ans))

with open('ans.pk', 'wb') as f:
    pickle.dump(submit_ans, f)
