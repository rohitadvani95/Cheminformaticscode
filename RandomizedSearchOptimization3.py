# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:37:20 2020

@author: kirby
"""
import numpy as np
import pandas as pd



from sklearn.model_selection import cross_val_score, cross_validate



from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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


rawdata = pd.read_csv('C:\Users\kirby\OneDrive\Documents\CompoundData\LateAugustMetaComb7.csv', skiprows=0)

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])

print(df)

answers = pd.DataFrame(data, columns = ['Skin'])


X = df
print(type(X))

#columns = answers

y = answers

print(type(y))



#X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=12, stratify=y)


#print('Train/Test Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#sss = StratifiedShuffleSplit(n_splits= 10, test_size=0.02, random_state= 10)

#clf = RandomForestClassifier()

#print(sss.get_n_splits(X, y))
#print(sss)

#print('Classifying Without Any Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=5)) ## It uses StratifiedKFold default
#print('Classifying With StratifiedShuffleSplit Cross Validation : ', cross_val_score(clf(), X_train, Y_train, cv=StratifiedShuffleSplit(n_splits=5)))

#split(X, y, groups=none)

NUM_TRIALS = 10


non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)


for i in range(NUM_TRIALS):

    sss_outer = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=10)
    sss_inner = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=10)

    param_grid = {"max_depth": [3, None],
              #"max_features": [1, 3, 10],
              "min_samples_split": [2, 5]}
              #"min_samples_split": [1, 3, 10]}
              #"classifier__min_samples_leaf": [1, 3, 10],
              # "bootstrap": [True, False],
              #"classifier__criterion": ["gini", "entropy"]}

    RFC = RandomForestClassifier(random_state = 10)

    estimator = RFC
    clf = GridSearchCV(estimator, param_grid=param_grid, cv=sss_inner)
    #print(estimator.get_params().keys())
    clf.fit(X, y)
    non_nested_scores[i] = clf.best_score_
    nested_score = cross_val_score(clf, X, y, cv = sss_outer)
    nested_scores[i] = nested_score.mean()
    
score_difference = non_nested_scores - nested_scores
print(non_nested_scores.mean())
print(nested_scores.mean())

#print(grid_search)


#calibrated_forest = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=10))
#print("3")

#grid_search.fit(X_train, y_train)

#estimator.get_params().keys()

#cross_val_score(clf, X, y, cv=sss_outer)


