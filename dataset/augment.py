'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Apply SMOTE or ADASYN to training data.
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Applies rolling for data augmentation.')

parser.add_argument('-i', type=str, metavar='source dir', default='dataset/partitioned/', help='source directory containing .npy files')
parser.add_argument('-o', type=str, metavar='save dir', default='dataset/augmented/', help='save directory for oversampled .npy files')
args = parser.parse_args()

load_dir = os.path.normpath(args.i).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN

# load dataset
x_train = np.load(load_dir + '/x_train.npy')
y_train = np.load(load_dir + '/y_train.npy')

# Augment data by 'rolling' the flux values over time
for i in np.where(y_train == 1)[0]:
    for j in range(128): # 128 is set semi-arbitrarily in order to match number of negatives
        r = np.random.randint(x_train.shape[0])
        x_train = np.concatenate((x_train, np.expand_dims(np.roll(x_train[i], r, axis=0), axis=0)))
        y_train = np.concatenate((y_train, np.array([[1]])))

# Save dataset as formatted .npy files
np.save(save_dir + '/x_train.npy', x_train)
np.save(save_dir + '/y_train.npy', np.expand_dims(y_train, 1))

print('Data augmented and saved.')
