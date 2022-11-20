from sklearn import datasets, svm, metrics
from sklearn import tree
import numpy as np
#import matplotlib.pyplot as plt
from utils import (
        preprocess_digits,
        train_dev_test_split,
        h_param_tuning,
        h_param_tuning_dec,
        data_viz,
        get_all_h_param_comb,
        predict
)


import argparse
from collections import Counter
import json
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str, required=True)
parser.add_argument('--random_state', default=1, help="Minimum count for tags in the dataset",
                                    type=int)
args = parser.parse_args()

print('input args clf_name is', args.clf_name, 'and random state is ', args.random_state)


#train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
dev_frac = args.random_state/100
test_frac = 0.1
train_frac = 1-(dev_frac+test_frac)
print('dev_frac is ', dev_frac)
print('test_frac is' , test_frac)
print('train_frac is ', train_frac)

#assert train_frac + dev_frac + test_frac == 1.0



# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {"gamma": gamma_list, "C": c_list}

h_param_comb = get_all_h_param_comb(params)

# PART: load dataset -- data from csv, tsv, jsonl, pickle

digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
del digits

models = [[tree.DecisionTreeClassifier(), 'decision_tree'], [svm.SVC(), 'svm']]

perf_test = {}

metric = metrics.accuracy_score
for model in models:
    ls = []
    for i in range(5):
        print(f"\nTraining for split: {i + 1}")
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)
        clf = model[0]
        if model[1] == 'svm':
            best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)

        else:
            best_model, best_metric, best_h_params = h_param_tuning_dec(clf, x_train, y_train, x_dev, y_dev, metric)

        ls.append(predict(best_model, x_test, y_test, metric))
        perf_test[model[1]] = ls

        ## Confusion Matrix

print("\n")
for i in range(5):
    print(i + 1, round(perf_test['svm'][i], 2), round(perf_test['decision_tree'][i], 2))
print("mean: ", round(np.mean(perf_test['svm']), 2), " ", round(np.mean(perf_test['decision_tree']), 2))
print("std: ", round(np.std(perf_test['svm']), 2), " ", round(np.std(perf_test['decision_tree']), 2))
