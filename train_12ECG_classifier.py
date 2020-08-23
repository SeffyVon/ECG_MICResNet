#!/usr/bin/env python

import numpy as np, os, sys

from signal_processing import read_signal
from train_NN_full import train_NN_sig_MIL_full
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

    headers = list()
    for i in range(num_files):
        header = load_challenge_header(header_files[i])
        headers.append(header)

    # Train model.
    print('Split dataset...')
    headers_datasets = get_dataset(headers)

    # make cwt
    print('Read signal ...')
    fDatas = read_signal(input_directory, headers_datasets, output_directory)

    assert len(fDatas) == len(headers)
    del headers, header_files, num_files

    # train and save the best model
    print('Training NN ... ')
    train_NN_sig_MIL_full(headers_datasets, output_directory, fDatas)

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