# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:37:20 2020

@author: kirby
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


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


rawdata = pd.read_csv('C:\Users\kirby\OneDrive\Documents\CompoundData\LateAugustMetaComb7.csv', skiprows=0)

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])

print(df)

answers = pd.DataFrame(data, columns = ['Skin'])


X = df
print(type(X))

columns = answers

y = columns

print(type(y))



X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=12, stratify=y)


print('Train/Test Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#sss = StratifiedShuffleSplit(n_splits= 10, test_size=0.02, random_state= 10)

#clf = RandomForestClassifier()

#print(sss.get_n_splits(X, y))
#print(sss)

#print('Classifying Without Any Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=5)) ## It uses StratifiedKFold default
#print('Classifying With StratifiedShuffleSplit Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=StratifiedShuffleSplit(n_splits=5)))


sss_outer = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=10)
sss_inner = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=10)
#pipe_logistic = ([('scl', StandardScaler()),('clf', RandomForestClassifier(penalty='l1'))])

#print("3")
calibrated_forest = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=10))
print("3")
param_grid = {'base_estimator__max_depth': [2, 4, 6, 8]}

#parameters = {'clf__C': logspace(-4,1,50)}
grid_search = GridSearchCV(estimator=calibrated_forest, param_grid=param_grid, verbose=1, scoring='f1', cv=sss_inner)
cross_val_score(grid_search, X, y, cv=sss_outer)



#grid = GridSearchCV(clf(random_state=123),
 #                   param_grid = {'max_depth': [None, 2,3,5], 'max_features' : ['auto','sqrt', 'log2'], 'n_estimators': [10,100],},

  #                  cv = sss(n_splits=5, random_state=123),
   #                 verbose=50,
    #                n_jobs=-1)
#grid.fit(X_train, Y_train)

#print('\nBest R^2 Score : %.2f'%grid.best_score_, ' Best Params : ', str(grid.best_params_))

