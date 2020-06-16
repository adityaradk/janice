'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Tests a Support Vector Machine classifier
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Test a Support Vector Machine classifier.')

parser.add_argument('-tdir', type=str, metavar='test set dir', default='dataset/normalized/', help='source directory containing .npy train files')
parser.add_argument('-mdir', type=str, metavar='model file dir', default='models/svm/files/model.pkl', help='save file for trained model')
args = parser.parse_args()

test_dir = os.path.normpath(args.tdir).replace('\\', '/')
model_dir = os.path.normpath(args.mdir).replace('\\', '/')

import numpy as np
import pickle as pkl

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

x = np.load(test_dir + '/x_test.npy')
y = np.load(test_dir + '/y_test.npy')

loaded_model = pkl.load(open(model_dir, 'rb'))

result = loaded_model.predict(x)

print(accuracy_score(result, y))
print(confusion_matrix(result, y))
