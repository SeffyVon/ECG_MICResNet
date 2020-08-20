#!/usr/bin/env python

import numpy as np, os, sys
from global_vars import labels, run_name
from get_12ECG_features import get_12ECG_features
from resnet1d import ECGResNet
import torch
from write_signal import filter_data, write_signal
import random

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

torch.manual_seed(0)
random.seed(0)
def get_sig(data):
    
    fData = filter_data(data[:12,:], highcut=50.0)
    
    return fData

def segment_sig(fData, mode='random'):
    j_sig = 0
    if mode == 'random':
        j_sig = random.randint(0, min(max(fData.shape[1] - 3000,1), 33000))
    else:
        j_sig = max(0, (fData.shape[1]-3000)//2) # center

    fData2 = fData[:,j_sig:j_sig+3000]
    if fData2.shape[1] != 3000:
            fData2 = np.pad(fData2, pad_width=((0,0),(0,3000-fData2.shape[1])), 
                mode='constant', constant_values=0)

    return fData2

def run_12ECG_classifier(data,header_data,loaded_model, mode='random'):

    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model['model']
    classes = loaded_model['classes']
    fData = get_sig(data)
    with torch.no_grad():
        model.eval()
        current_score = None
        if mode == 'random':
            sig_tensor = torch.from_numpy(np.array([segment_sig(fData, mode) for _ in range(21)])).type(torch.FloatTensor)
            outputs = model(sig_tensor.to(device))
            current_score = np.max(torch.sigmoid(outputs).cpu().numpy(), axis=0)

        else: # center
            sig_tensor = torch.from_numpy(np.array([segment_sig(fData, mode)])).type(torch.FloatTensor)
            #print(sig_tensor.shape)
            outputs = model(sig_tensor.to(device))
            current_score = torch.sigmoid(outputs).cpu().numpy()[0]     
        current_label = np.round(current_score).astype(int)
    
        return current_label, current_score, classes

def load_trained_model(model_saved_path):
    model = ECGResNet(12, len(labels)).to(device)
    # load saved model
    model.load_state_dict(torch.load(model_saved_path+'/{}_model.dict'.format(run_name)))#, map_location=torch.device('cpu')))
    return model

def load_12ECG_model(input_directory):
    # load the model from disk 
    loaded_model = {}
    loaded_model['model'] = load_trained_model(input_directory)
    loaded_model['classes'] = [str(l) for l in labels]
    return loaded_model