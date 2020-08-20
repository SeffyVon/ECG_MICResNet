#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from write_signal import write_signal
from train_NN_sig_MIL import train_NN_sig_MIL
from manipulations import get_dataset

def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')

    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    num_files = len(header_files)

    recordings = list()
    headers = list()
    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording)
        headers.append(header)

    # Train model.
    print('Training and saving model...')
    headers_datasets, recordings_datasets = get_dataset(headers, recordings)

    # make cwt
    print('Write signal ...')
    write_signal(recordings_datasets, headers_datasets, output_directory)

    del recordings_datasets 
    del headers, header_files, num_files

    # train and save the best model
    print('Training NN ... ')
    #train_NN_sig_feature(headers_datasets, output_directory, features)
    train_NN_sig_MIL(headers_datasets, output_directory)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Load challenge data.
def load_challenge_header(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    return header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)