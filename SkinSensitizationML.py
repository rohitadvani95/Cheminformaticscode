import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


# Import packages to do the classifying

from sklearn.svm import SVC

import csv

with open(r'C:\Users\kirby\OneDrive\Documents\LateAugustMetaComb4.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)

rawdata = pd.read_csv('C:\Users\kirby\OneDrive\Documents\LateAugustMetaComb4.csv', skiprows=0)
#rawdata.drop(["GNPS"], axis = 1, inplace = True)

print(rawdata)

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])
print(df)

answers = pd.DataFrame(data, columns = ['Skin'])
#print(answers)

X = df

y = answers

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.22, random_state=10)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#classifier = SVC(kernel = 'linear', random_state = 100, C = 1000)

#classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 10, C = 5)

classifier = RandomForestClassifier(n_estimators = 45, criterion = 'entropy', random_state = 980, warm_start= True)

#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 990)
classifier.fit(X_train, y_train)
Y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, Y_pred)

print(cm)







