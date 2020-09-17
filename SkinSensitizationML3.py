# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:31:53 2020

@author: kirby
"""

import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.svm import SVC
import numpy as np

# Import packages to do the classifying


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
#rawdata.drop(["GNPS"], axis = 1, inplace = True)

#print(rawdata)

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])
#df = pd.DataFrame(data, columns = ['MW'])
print(df)

answers = pd.DataFrame(data, columns = ['Skin'])
#print(answers)

X = df
print(type(X))

columns = answers

y = columns

print(type(y))

#y = y.reindex(columns=columns)
#y[columns] = y[columns].astype(int)



#X = np.ones((107,1))
#y = np.hstack(([0] * 91, [1] * 16))

#print(X)
#print(y)

#skf = StratifiedKFold(n_splits = 10, random_state=None, shuffle=False)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.12, random_state=None)

print(sss.get_n_splits(X, y))
print(sss)

#for train, test in skf.split(X, y):
 #   print('train -  {}   |   test -  {}'.format(
  #          np.bincount(y[train]), np.bincount(y[test])))



for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train = X.iloc[train_index] 
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index] 
    y_test = y.iloc[test_index]
print(X_train.shape)
print(X_test.shape)    
print(y_train.shape)    
print(y_test.shape)
    




#clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 100) 

#clf = RandomForestClassifier(n_estimators = 740, criterion = 'entropy', random_state = 0, warm_start= True)

#clf = SVC(kernel = 'rbf', random_state = 10, gamma = 1000, C = 0.1)

clf = SVC(kernel = 'linear', random_state = 10, C = 0.3)



scores = cross_val_score(clf, X, y, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





