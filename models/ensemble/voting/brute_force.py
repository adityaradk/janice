'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Brute forces every possible combination of models to create all possible voting classifiers.
Note that this file does not have arguments.
'''

import numpy as np
import pickle as pkl
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from statistics import mode
from itertools import combinations

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Load validation data
x = np.load('dataset/partitioned/x_val.npy')

y = np.load('dataset/partitioned/y_val.npy').flatten().tolist()

adaboost = pkl.load(open('models/adaboost/files/model.pkl', 'rb'))
o_adaboost = adaboost.predict(x).flatten().tolist()

knn = pkl.load(open('models/knn/files/model.pkl', 'rb'))
o_knn = knn.predict(x).flatten().tolist()

naive_bayes = pkl.load(open('models/naive_bayes/files/model.pkl', 'rb'))
o_naive_bayes = naive_bayes.predict(x).flatten().tolist()

random_forest = pkl.load(open('models/random_forest/files/model.pkl', 'rb'))
o_random_forest = random_forest.predict(x).flatten().tolist()

svm = pkl.load(open('models/svm/files/model.pkl', 'rb'))
o_svm = svm.predict(x).flatten().tolist()

mlp = load_model('models/mlp/files/model.h5')
o_mlp = mlp.predict_on_batch(x).numpy().flatten()

cnn = load_model('models/cnn/files/model.h5')
o_cnn = cnn.predict_on_batch(np.expand_dims(x, 2)).numpy().flatten()

o_mlp = [1 if o_mlp_ > 0.95 else 0 for o_mlp_ in o_mlp]
o_cnn = [1 if o_cnn_ > 0.95 else 0 for o_cnn_ in o_cnn]
o_naive_bayes = [1 if o_naive_bayes_ > 0.3 else 0 for o_naive_bayes_ in o_naive_bayes]

#1+ 5 + 10 + 10

model_names = ['adaboost', 'knn', 'naive_bayes', 'random_forest', 'svm', 'mlp', 'cnn']
preds = {'adaboost' : o_adaboost, 'knn' : o_knn, 'naive_bayes' : o_naive_bayes, 'random_forest' : o_random_forest, 'svm' : o_svm, 'mlp' : o_mlp, 'cnn' : o_cnn}
results, n_done = [], 0

for i in range(2, 6):
    for j in combinations(model_names, i):
        print('Evaluating combination ' + str(n_done) + '.       ', end='\r')
        tp, tn, p, n = 0, 0, 0, 0
        for k in range(len(y)):
            votes = []
            for l in j:
                votes.append(preds[l][k])
            try:
                pred = mode(votes)
            except:
                pred = 1
            if y[k] == 1:
                if pred == y[k]:
                    tp += 1
                p += 1
            elif y[k] == 0:
                if pred == y[k]:
                    tn += 1
                n += 1
        results.append([((tp/p)+(tn/n))/2, list(j)])
        n_done += 1

results.sort()
results.reverse()

top = []

for i in results:
    if i[0] == results[0][0]:
        top.append(i[1])

top.sort(key=len)

print('\nTOP PERFORMERS:\n')
top_a = None
for i in top:
    print(i)
    top_a = np.append(top_a, i) if top_a is not None else i

print('\nMOST COMMON CLASSIFIERS:\n')
print(np.unique(top_a, return_counts=True))
