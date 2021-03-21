#!/user/bin/env python
# -*- coding:utf-8 -*-
import pickle
import numpy as np
import pandas as pd


def load(phase):
    explicit = pd.read_csv('data/' + phase + '_explicit.csv',header=None).to_numpy()
    implicit = pd.read_csv('data/' + phase + '_implicit.csv',header=None).to_numpy()
    if phase!='test':
        tag = pd.read_csv('data/' + phase + '_tag.csv',header=None).to_numpy().ravel()
        return explicit, implicit, tag
    else:
        return explicit,implicit


def get_slot_set():
    slot_set = []
    with open('./dataset/symptom.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip())
    return slot_set


def get_disease_set():
    total_disease = []
    with open('./dataset/disease.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            total_disease.append(line.strip())
    return total_disease


def load_data():
    slot_set = get_slot_set()
    goals = {}
    with open('./dataset/train.pk', 'rb') as f:
        goals['train'] = pickle.load(f)

    with open('./dataset/dev.pk', 'rb') as f:
        goals['dev'] = pickle.load(f)

    with open('./dataset/test.pk', 'rb') as f:
        goals['test'] = pickle.load(f)

    total_disease = get_disease_set()
    slot2num(goals['train'], slot_set, total_disease, 'train')
    slot2num(goals['dev'], slot_set, total_disease, 'dev')
    slot2num(goals['test'], slot_set, total_disease, 'test')


def slot2num(data, slot_set, disease_set, phase):
    # true:1 false:-1 not mention:0
    slot_dict = {v: k for k, v in enumerate(slot_set)}
    disease_dict = {v: k for k, v in enumerate(disease_set)}
    explicit = np.zeros((len(data), len(slot_set)))
    implicit = np.zeros((len(data), len(slot_set)))
    tags = np.zeros((len(data), 1))
    for i, sample in enumerate(data):
        symptom = []
        for slot, exist in sample['explicit_inform_slots'].items():
            explicit[i][slot_dict[slot]] = 1 if exist else -1
        for slot, exist in sample['implicit_inform_slots'].items():
            implicit[i][slot_dict[slot]] = 1 if exist else -1
            symptom.append(slot_dict[slot])
        if phase != 'test':
            tags[i] = disease_dict[sample['disease_tag']]
    np.savetxt('data/' + phase + '_explicit.csv', explicit, fmt='%d', delimiter=',')
    np.savetxt('data/' + phase + '_implicit.csv', implicit, fmt='%d', delimiter=',')
    if phase != 'test':
        np.savetxt('data/' + phase + '_tag.csv', tags, fmt='%d', delimiter=',')
    return explicit, implicit, tags


if __name__ == '__main__':
    load_data()
    load("train")
