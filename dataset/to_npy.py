'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Converts CSV dataset to .npy files with train/test inputs and labels.

Dataset obtained from:
https://github.com/winterdelta/KeplerAI
Dataset was created by user WinterDelta (WΔ)
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Converts CSV dataset to .npy files.')

parser.add_argument('-i', type=str, metavar='source dir', default='dataset/raw/', help='source directory containing exotrain and exotest csv files')
parser.add_argument('-o', type=str, metavar='save dir', default='dataset/raw/', help='save directory for generated .npy files')
args = parser.parse_args()

load_dir = os.path.normpath(args.i).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

import numpy as np
import pandas as pd

# Load csv data into pandas dataframes
df_train = pd.read_csv(load_dir + '/exoTrain.csv')
df_test = pd.read_csv(load_dir + '/exoTest.csv')

# Load targets into numpy arrays and change labels from 2/1 to 1/0
y_train = np.expand_dims(df_train.LABEL.to_numpy() - 1, axis=1)
y_test = np.expand_dims(df_test.LABEL.to_numpy() - 1, axis=1)

# Load flux readings into numpy arrays
x_train = df_train.drop(['LABEL'], 1).to_numpy()
x_test = df_test.drop(['LABEL'], 1).to_numpy()

# Save dataset as formatted .npy files
np.save(save_dir + '/x_train.npy', x_train)
np.save(save_dir + '/x_test.npy', x_test)
np.save(save_dir + '/y_train.npy', y_train)
np.save(save_dir + '/y_test.npy', y_test)

print('Dataset saved with shapes ' + str((x_train.shape, x_test.shape, y_train.shape, y_test.shape))[1:-1])
