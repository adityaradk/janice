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
parser = argparse.ArgumentParser(description='Applies SMOTE or ADASYN.')

parser.add_argument('-ost', type=str, metavar='os technique', choices=['smote', 'adasyn'], default='smote', help='oversampling technique - smote or adasyn')
parser.add_argument('-i', type=str, metavar='source dir', default='dataset/partitioned/', help='source directory containing .npy files')
parser.add_argument('-o', type=str, metavar='save dir', default='dataset/oversampled/', help='save directory for oversampled .npy files')
args = parser.parse_args()

load_dir = os.path.normpath(args.i).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

os_technique = args.ost.lower()

import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN

# load dataset
x_train = np.load(load_dir + '/x_train.npy')
y_train = np.load(load_dir + '/y_train.npy')

# Oversample
oversample = SMOTE() if os_technique == 'smote' else ADASYN()
x_train_oversampled, y_train_oversampled = oversample.fit_resample(x_train, y_train)

# Save dataset as formatted .npy files
np.save(save_dir + '/' + os_technique + '/x_train.npy', x_train_oversampled)
np.save(save_dir + '/' + os_technique + '/y_train.npy', np.expand_dims(y_train_oversampled, 1))

print('Data oversampled with ' + os_technique.upper() + ' and saved.')
