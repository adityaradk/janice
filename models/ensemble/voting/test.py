'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Tests a voting classifier.
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Tests a voting classifier.')

parser.add_argument('-models', type=str, metavar='ensemble models', default='[\'adaboost\', \'cnn\', \'mlp\', \'random_forest\']', help='save file for trained model')
parser.add_argument('-tdir', type=str, metavar='train set dir', default='dataset/augmented/', help='source directory containing .npy train files')
args = parser.parse_args()

load_dir = os.path.normpath(args.tdir).replace('\\', '/')
model_names = args.models.strip('\'][\'').split('\', \'')

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

# Load test data

x = np.load('dataset/normalized/x_test.npy')

y = np.load('dataset/normalized/y_test.npy').flatten().tolist()

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

preds = {'adaboost' : o_adaboost, 'knn' : o_knn, 'naive_bayes' : o_naive_bayes, 'random_forest' : o_random_forest, 'svm' : o_svm, 'mlp' : o_mlp, 'cnn' : o_cnn}

tp, tn, p, n = 0, 0, 0, 0

for i in range(len(y)):
    votes = []
    for j in model_names:
        votes.append(preds[j][i])
    '''
    try:
        pred = mode(votes)
    except:
        pred = 1
    '''
    pred = 1 if votes.count(1) >= 2 else 0
    if y[i] == 1:
        print(y[i], votes)
        if pred == y[i]:
            tp += 1
        p += 1
    elif y[i] == 0:
        if pred == y[i]:
            tn += 1
        n += 1

# Calculate recall and precision
recall = tp/(tp + (n - tn))
precision = tp/(tp + (p - tp))

# Display results
print('\n--------------BEGIN TEST RESULTS--------------\n')

print('Sensitivity (TP Rate) is ' + str(round(tp/p , 2)))
print('Specificity (TN Rate) is ' + str(round(tn/n , 2)))
print('Precision is ' + str(round(precision, 2)))
print('Recall is ' + str(round(recall, 2)))
print('F1 Score is ' + str(round(2 * precision * recall / (precision + recall), 2)))

print('\n---------------END TEST RESULTS---------------\n')
