'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Trains a Support Vector Machine classifier using Scikit-Learn
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train a Support Vector Machine classifier.')

parser.add_argument('-pca', type=str, metavar='use PCA?', choices=['True', 'False'], default='False', help='boolean specifying if PCA should be used')
parser.add_argument('-tdir', type=str, metavar='train set dir', default='dataset/oversampled/smote/', help='source directory containing .npy train files')
parser.add_argument('-vdir', type=str, metavar='val set dir', default='dataset/partitioned/', help='source directory containing .npy train files')
parser.add_argument('-o', type=str, metavar='save file', default='models/svm/files/model.pkl', help='save file for trained model')
args = parser.parse_args()

load_dir = os.path.normpath(args.tdir).replace('\\', '/')
val_dir = os.path.normpath(args.vdir).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')
use_pca = eval(os.path.normpath(args.pca))

import numpy as np
import pickle as pkl

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load training data
x = np.load(load_dir + '/x_train.npy')
y = np.load(load_dir + '/y_train.npy')

# Load validation data
x_val = np.load(val_dir + '/x_val.npy')
y_val = np.load(val_dir + '/y_val.npy')

if use_pca:
    from sklearn.decomposition import PCA
    pca = PCA(0.80, random_state=0)
    x_train_pca = pca.fit_transform(x)
    x_val_pca = pca.transform(x_val)
    print(pca.components_.shape)


classifier = SVC(kernel='sigmoid', random_state=0)
classifier.fit(x, y)
y_pred = classifier.predict(x_val)

print(accuracy_score(y_pred, y_val))
print(confusion_matrix(y_pred, y_val))

pkl.dump(classifier, open(save_dir, 'wb'))
print('Model saved to ' + save_dir)
