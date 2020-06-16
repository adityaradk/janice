'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Trains a MultiLayer Perceptron (MLP) with tf.keras
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train a MultiLayer Perceptron (MLP).')

parser.add_argument('-tdir', type=str, metavar='train set dir', default='dataset/augmented/', help='source directory containing .npy train files')
parser.add_argument('-vdir', type=str, metavar='val set dir', default='dataset/partitioned/', help='source directory containing .npy train files')
parser.add_argument('-o', type=str, metavar='save file', default='models/mlp/files/model.h5', help='save file for trained model')
args = parser.parse_args()

load_dir = os.path.normpath(args.tdir).replace('\\', '/')
val_dir = os.path.normpath(args.vdir).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential()
model.add(Input((3197,)))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Load training data
x = np.load(load_dir + '/x_train.npy')
y = np.load(load_dir + '/y_train.npy')

# Load validation data
x_val = np.load(val_dir + '/x_val_k.npy')
y_val = np.load(val_dir + '/y_val_k.npy')

# Use ModelCheckpoint and EarlyStopping
checkpoint = ModelCheckpoint(save_dir, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

# Fit the model to the training data, first with a lower learning rate and no early stopping
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5))
model.fit(x, y, epochs=25, batch_size=32, verbose=1)

# Fit the model to the train set, now with early stopping
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
model.fit(x, y, epochs=75, batch_size=32, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpoint, early])

# Display validation results
print('\n-----------BEGIN VALIDATION RESULTS-----------\n')

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

# Load best model
model = load_model(save_dir)

# Get validation set without oversampling
x_val = np.load(val_dir + '/x_val.npy')
y_val = np.load(val_dir + '/y_val.npy').flatten()

preds = model.predict_on_batch(x_val).numpy().flatten()

auc = roc_auc_score(y_val, preds)
fpr, tpr, thresholds = roc_curve(y_val, preds)
print('ROC AUC Score is %.3f' % auc)

# Compute and display optimal threshold
# Threshold optimized to maximize tpr - fpr, with an arbitrary 0.05 reduction to reduce FN rate
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx] - 0.05
print("Optimal threshold is " + str(round(optimal_threshold, 2)))

# Plot the ROC curve for the model
plt.figure(num='ROC Curve for Validation Set')
plt.plot(fpr, tpr, linestyle='--', label='Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Compute accuracies
tp, tn, p, n = 0, 0, 0, 0
p_preds, n_preds = [], []
for i in range(len(y_val)):
    if y_val[i] == 1:
        if preds[i] >= optimal_threshold:
            tp += 1
        p_preds.append(preds[i])
        p += 1
    else:
        if preds[i] < optimal_threshold:
            tn += 1
        n_preds.append(preds[i])
        n += 1

# Calculate recall and precision
recall = tp/(tp + (n - tn))
precision = tp/(tp + (p - tp))

# Display stats
print('Sensitivity (TP Rate) is ' + str(round(tp/p , 2)))
print('Specificity (TN Rate) is ' + str(round(tn/n , 2)))
print('Precision is ' + str(round(precision, 2)))
print('Recall is ' + str(round(recall, 2)))
print('F1 Score is ' + str(round(2 * precision * recall / (precision + recall), 2)))

# Plot predictions
plt.figure(num='Predictions on Validation Set')
plt.plot(p_preds, 'r.')
plt.plot([None] * len(p_preds) + n_preds, 'b.')
plt.axhline(y=optimal_threshold, color='r', linestyle='--')
plt.show()

print('\n------------END VALIDATION RESULTS------------\n')
