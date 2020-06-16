'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Trains a bagging ensemble.
Note that this file possesses no arguments.
'''

import numpy as np
x_train = np.load('dataset/oversampled/smote/x_train.npy')
x_test = np.load('dataset/normalized/x_test.npy')
x_val = np.load('dataset/partitioned/x_val.npy')
y_train = np.load('dataset/oversampled/smote/y_train.npy')
y_test = np.load('dataset/normalized/y_test.npy')
y_val = np.load('dataset/partitioned/y_val.npy')


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

base_cls = DecisionTreeClassifier()
num_trees = 50
model = BaggingClassifier(base_estimator = base_cls,
                          n_estimators = num_trees,
                          random_state = 0)

model.fit(x_train,y_train)

y_pred = model.predict(x_val)

print(confusion_matrix(y_val,y_pred))

y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))
