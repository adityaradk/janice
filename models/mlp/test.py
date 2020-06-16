'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Tests a MultiLayer Perceptron (MLP) with tf.keras
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Test a MultiLayer Perceptron (MLP).')

parser.add_argument('-t', type=str, metavar='threshold', default=0.5, help='classification threshold')
parser.add_argument('-tdir', type=str, metavar='test set dir', default='dataset/normalized/', help='source directory containing .npy train files')
parser.add_argument('-mdir', type=str, metavar='model file dir', default='models/mlp/files/model.h5', help='save file for trained model')
args = parser.parse_args()

threshold = float(args.t)

test_dir = os.path.normpath(args.tdir).replace('\\', '/')
model_dir = os.path.normpath(args.mdir).replace('\\', '/')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load test data
x = np.load(test_dir + '/x_test.npy')
y = np.load(test_dir + '/y_test.npy').flatten()

# Load model and generate predictions
model = load_model(model_dir)
o = model.predict_on_batch(x).numpy().flatten()

tp, tn, p, n = 0, 0, 0, 0
p_preds, n_preds = [], []
for i in range(len(y)):
    if y[i] == 1:
        if o[i] >= threshold:
            tp += 1
        p_preds.append(o[i])
        p += 1
    else:
        if o[i] < threshold:
            tn += 1
        n_preds.append(o[i])
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

import matplotlib.pyplot as plt

plt.figure(num='Predictions on Test Set')
plt.plot(p_preds, 'r.')
plt.plot([None] * len(p_preds) + n_preds, 'b.')
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

print('\n---------------END TEST RESULTS---------------\n')
