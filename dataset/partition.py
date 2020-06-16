'''
  █ ▄▀█ █▄ █ █ █▀▀ █▀▀
█▄█ █▀█ █ ▀█ █ █▄▄ ██▄
Just ANother Intelligent Classifier for Exoplanets

Extracts and removes validation set from training set (normalized data, by default).
'''

import os
import sys
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description='Extracts and removes validation set from training set.')

parser.add_argument('-f', type=str, metavar='fraction', default='0.1', help='fraction of positive/negative samples to be taken from train to val')
parser.add_argument('-i', type=str, metavar='source dir', default='dataset/normalized/', help='source directory containing train set .npy file')
parser.add_argument('-o', type=str, metavar='save dir', default='dataset/partitioned/', help='save directory for generated .npy files')
args = parser.parse_args()

f = float(os.path.normpath(args.f))
load_dir = os.path.normpath(args.i).replace('\\', '/')
save_dir = os.path.normpath(args.o).replace('\\', '/')

import random
import numpy as np

# Load normalized data
x_train = np.load(load_dir + '/x_train.npy')
y_train = np.load(load_dir + '/y_train.npy')


# Find indices to put in validation set
p_indices = np.where(y_train == 1)[0].tolist()
val_p_indices = random.sample(p_indices, int(len(p_indices) * f))
n_indices = np.where(y_train == 0)[0].tolist()
val_n_indices = random.sample(n_indices, int(len(n_indices) * f))

# Create positive and negative parts of the validation set
x_val_p = x_train[val_p_indices]
y_val_p = y_train[val_p_indices]
x_val_n = x_train[val_n_indices]
y_val_n = y_train[val_n_indices]

# Remove validation indices from train set
x_train = np.delete(x_train, val_p_indices + val_n_indices, axis=0)
y_train = np.delete(y_train, val_p_indices + val_n_indices, axis=0)

# Combine positives and negatives
x_val = np.concatenate((x_val_p, x_val_n))
y_val = np.concatenate((y_val_p, y_val_n))

# Oversample positives for use with Keras
x_val_k, y_val_k = x_val_p, y_val_p
while len(x_val_k) < len(x_val_n):
    x_val_k = np.concatenate((x_val_k, x_val_p))
    y_val_k = np.concatenate((y_val_k, y_val_p))
x_val_k = np.concatenate((x_val_k, x_val_n))
y_val_k = np.concatenate((y_val_k, y_val_n))


# Save dataset as formatted .npy files
np.save(save_dir + '/x_train.npy', x_train)
np.save(save_dir + '/x_val_k.npy', x_val_k)
np.save(save_dir + '/x_val.npy', x_val)
np.save(save_dir + '/y_train.npy', y_train)
np.save(save_dir + '/y_val_k.npy', y_val_k)
np.save(save_dir + '/y_val.npy', y_val)

print('Train set partitioned.')
