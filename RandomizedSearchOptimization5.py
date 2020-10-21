# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:32:43 2020

@author: kirby
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats


from sklearn.exceptions import FitFailedWarning
import warnings


from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, cross_validate


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

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
#X = df
#print(type(X))

#columns = answers

#y = answers

#print(type(y))



#X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=12, stratify=y)


#print('Train/Test Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#sss = StratifiedShuffleSplit(n_splits= 10, test_size=0.02, random_state= 10)

#clf = RandomForestClassifier()

#print(sss.get_n_splits(X, y))
#print(sss)

#print('Classifying Without Any Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=5)) ## It uses StratifiedKFold default
#print('Classifying With StratifiedShuffleSplit Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=StratifiedShuffleSplit(n_splits=5)))

#split(X, y, groups=none)

#columns = ['MW', 'SA', 'LogP']

#df = df[columns]
#print(df.head())

features = df
clss = np.asarray(answers)

print(type(features))
print(type(clss))



n_splits = 10
test_size = 0.2

accuracies_train = np.empty(n_splits, np.float32)
accuracies_test = np.empty(n_splits, np.float32)
average_precisions_train = np.empty(n_splits, np.float32)
average_precisions_test = np.empty(n_splits, np.float32)

interval = np.linspace(0, 1, 101, dtype=np.float32)

precisions_train = np.empty(n_splits, np.float32)
precisions_test = np.empty(n_splits, np.float32)
feature_importances = np.zeros((n_splits, len(features.columns.values)), np.float32)
 
                               

best_params = []


#NUM_TRIALS = 10

for i, (train_index, test_index) in enumerate(
        StratifiedShuffleSplit(n_splits, test_size, random_state = 42).split(
                features.values, clss)):
    features_train, features_test = (features.values[train_index], 
                                     features.values[test_index])
    clss_train, clss_test = clss[train_index], clss[test_index]

    RFC = RandomForestClassifier
    n_trees = 100
    classifier = GridSearchCV(
        Pipeline([('feature_selection', feature_selection.SelectKBest(k = all(features_train)),
                  ('classify', RFC(
                      n_trees, random_state=42,
                      class_weight='balanced_subsample')))]),
        param_grid={
            'feature_selection__k': np.arange(5, 51, 5),
            'classify__max_depth': np.arange(3, 8, 1),
            'classify__min_samples_leaf': np.arange(1, 4, 1)},
        n_jobs=-1,
        cv=StratifiedShuffleSplit(n_splits, test_size, random_state=42))
    
    
    #Pipeline.get_params(features, deep= True)
    #kbest = SelectKBest()
    #kbest.fit_transform(features, clss)
    classifier.fit(features_train, clss_train)
    pred_train = classifier.predict_proba(features_train)[:, 1]
    pred_test = classifier.predict_proba(features_test)[:, 1]
    
    #compute metrics
    accuracies_train[i] = balanced_accuracy_score(
            clss_train, np.asarray(pred_train > 0.5, np.int))
    accuracies_test[i] = balanced_accuracy_score(
            clss_test, np.asarray(pred_test > 0.5, np.int))
    average_precisions_train[i] = average_precision_score(
            clss_test, pred_train) 
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
    selected_features_mask = (classifier.best_estimator_
                              .named_steps['feature_selection']
                              .get_support())
    feat_imp = permutation_importance(
            classifier.best_estimator_.named_steps['classify'],
            features.values[:, selected_features_mask],
            clss, n_jobs=-1, random_state=42)
    feature_importances[i, selected_features_mask] += \
        feat_imp.importances_mean
        
    best_params.append((
            classifier.best_estimator_.named_steps['feature_selection'].k,
            classifier.best_estimator_.named_steps['classify'].max_depth,
            classifier.best_estimator_.named_steps['classify'].min_samples_leaf))
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

print(stats)
final_params = scipy.stats.mode(best_params)[0][0]
print(f'Optimal number of features (mode): {final_params[0]}')
print(f'Optimal number of maximum depth (mode): {final_params[1]}')
print(f'Optimal number of minimum leaf samples (mode): {final_params[2]}')








