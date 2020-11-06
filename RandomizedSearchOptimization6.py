# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:50:40 2020

@author: kirby
"""


import os

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats

from warnings import simplefilter

simplefilter(action='ignore', category = FutureWarning)



from sklearn.exceptions import FitFailedWarning
import warnings


from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, cross_validate


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
#from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
 


import csv

with open(r'C:\Users\kirby\OneDrive\Documents\CompoundData\LateAugustMetaComb7.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)


rawdata = pd.read_csv(r'C:\Users\kirby\OneDrive\Documents\CompoundData\LateAugustMetaComb7.csv', skiprows=0)

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])

print(df)

answers = pd.DataFrame(data, columns = ['Skin'])

print(answers)


features = df
clss = pd.to_numeric(answers['Skin']).values


print(type(features))
print(type(clss))



n_splits = 10
test_size = 0.2

accuracies_train = np.empty(n_splits, np.float32)
accuracies_test = np.empty(n_splits, np.float32)
average_precisions_train = np.empty(n_splits, np.float32)
average_precisions_test = np.empty(n_splits, np.float32)

interval = np.linspace(0, 1, 101, dtype=np.float32)

precisions_train = np.empty((n_splits, 101), np.float32)
precisions_test = np.empty((n_splits, 101), np.float32)
feature_importances = np.zeros((n_splits, len(features.columns.values)), np.float32)
 
                               

best_params = []


#NUM_TRIALS = 10

for i, (train_index, test_index) in enumerate(
        StratifiedShuffleSplit(n_splits, test_size, random_state = 42).split(
                features.values, clss)):
    features_train, features_test = (features.values[train_index], 
                                     features.values[test_index])
    clss_train, clss_test = clss[train_index], clss[test_index]
    
    print(clss_train)
    
    n_trees = 100
    
    
    param_grid = {'max_depth': [3, None],
    #'classify__max_depth': np.arange(3, 8, 1),
    'min_samples_leaf': np.arange(1, 4, 1),
    #"max_features": [1, 3, 10],
    'min_samples_split': [2, 5]}
              #"min_samples_split": [1, 3, 10]}
              #"classifier__min_samples_leaf": [1, 3, 10],
              # "bootstrap": [True, False],
              #"classifier__criterion": ["gini", "entropy"]}

    RFC = RandomForestClassifier(random_state = 10)
    
    
    estimator = RFC
    
    estimator = estimator.fit(features_train, clss_train.ravel())
    
    print(estimator.get_params().keys())
    classifier = GridSearchCV(estimator, param_grid=param_grid, refit=True, cv=StratifiedShuffleSplit(n_splits, test_size, random_state=42))
    #print(estimator.get_params().keys())
    
    
    print(features.shape)
    print(clss.shape)
    print(classifier.get_params().keys())
    
    
    classifier.fit(features, clss.ravel())
    
    #Pipeline.get_params(features, deep= True)
    #kbest = SelectKBest()
    #kbest.fit_transform(features, clss)
    
    pred_train = classifier.predict_proba(features_train)[:, 1]
    pred_test = classifier.predict_proba(features_test)[:, 1]
    
    print(pred_train)
    print(pred_test)
    
    #compute metrics
    accuracies_train[i] = balanced_accuracy_score(
            clss_train, np.asarray(pred_train > 0.5, np.int))
    
    print(accuracies_train[i])
    
    accuracies_test[i] = balanced_accuracy_score(
            clss_test, np.asarray(pred_test > 0.5, np.int))
    average_precisions_train[i] = average_precision_score(
            clss_train, pred_train) 
    average_precisions_test[i] = average_precision_score(
            clss_test, pred_test)
    precision_train, recall_train, _ = precision_recall_curve(
            clss_train, pred_train)
    precisions_train[i] = np.interp(
            interval, recall_train[::-1], precision_train[::-1])
    precision_test, recall_test, _ = precision_recall_curve(
            clss_test, pred_test)
    precisions_test[i] = np.interp(
            interval, recall_test[::-1], precision_test[::-1])
    selected_features_mask = (classifier.best_estimator_.named_steps['feature_selection'].get_support())
    feat_imp = permutation_importance(
        classifier.best_estimator_.named_steps['classify'], 
        features.values[:, selected_features_mask], 
        clss, n_jobs = -1, random_state = 42)
    feature_importances[i, selected_features_mask] += \
        feat_imp.importances_mean
        
    best_params.append((
            classifier.best_estimator_['classify'].max_depth,
            classifier.best_estimator_['classify'].min_samples_leaf))
print(best_params)

stats = {'accuracy_train': np.mean(accuracies_train),
         'accuracy_std_train': np.std(accuracies_train),
         'average_precision_train': np.mean(average_precisions_train),
         'average_precision_std_train': np.std(average_precisions_train),
         'precision_mean_train': np.mean(precisions_train, axis=0),
         'precision_std_train': np.std(precisions_train, axis=0),
        
         'accuracy_test': np.mean(accuracies_test),
         'accuracy_std_test': np.std(accuracies_test),
         'average_precision_test': np.mean(average_precisions_test),
         'average_precision_std_test': np.std(average_precisions_test),
         'precision_mean_test': np.mean(precisions_test, axis=0),
         'precision_std_test': np.std(precisions_test, axis=0),
         
         'feature_importances': np.mean(feature_importances, axis=0),
         'feature_importances_std': np.std(feature_importances, axis=0)}

print("6")
print(stats)
print("2")

final_params = scipy.stats.mode(best_params)[0][0]

print("6")

print(f'Optimal number of maximum depth (mode): {final_params[1]}')
print(f'Optimal number of minimum leaf samples (mode): {final_params[2]}')

