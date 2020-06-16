'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Normalize data.
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Normalizes data.')

parser.add_argument('-i', type=str, metavar='source dir', default='dataset/raw/', help='source directory containing .npy files')
parser.add_argument('-o', type=str, metavar='save dir', default='dataset/normalized/', help='save directory for normalized .npy files')
args = parser.parse_args()

load_dir = os.path.normpath(args.i).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

import numpy as np
from sklearn.preprocessing import normalize

# load dataset
x_train = np.load(load_dir + '/x_train.npy')
x_test = np.load(load_dir + '/x_test.npy')
y_train = np.load(load_dir + '/y_train.npy')
y_test = np.load(load_dir + '/y_test.npy')

# Normalize dataset
x_train_normalized, x_test_normalized = normalize(x_train), normalize(x_test)

# Save dataset as formatted .npy files
np.save(save_dir + '/x_train.npy', x_train_normalized)
np.save(save_dir + '/x_test.npy', x_test_normalized)
np.save(save_dir + '/y_train.npy', y_train.copy())
np.save(save_dir + '/y_test.npy', y_test.copy())

print('Data normalized and saved.')
