import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import preprocessing
from numpy.random import RandomState

seed = RandomState()

def loadData():
    data = []

    # read training data
    f = open('spambase.data')
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        data.append(row)
    f.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=seed)
    
    return X_train, X_test, y_train, y_test




# set up data
X_train, X_test, y_train, y_test = loadData()
m, n = X_train.shape

# standardize training and testing sets
standardize = preprocessing.StandardScaler().fit(X_train)
X_train = standardize.transform(X_train)
X_test = standardize.transform(X_test)

# train logistic regression model
c = 1.0 # fitting parameter (inverse regularization)
LRClassifier = LogisticRegression(C=c, random_state=seed).fit(X_train, y_train)

# evaluate model
meanAccuracy = LRClassifier.score(X_test, y_test) # data is skewed so not the most representative metric
y_pred = LRClassifier.predict(X_test)
f1Score = f1_score(y_test, y_pred)

print('mean accuracy: ', meanAccuracy)
print('f1 score: ', f1Score)