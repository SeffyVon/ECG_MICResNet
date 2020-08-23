#!/usr/bin/env python

import numpy as np, os, sys
from global_vars import labels, run_name, n_segments, max_segment_len
from get_12ECG_features import get_12ECG_features
from resnet1d import ECGBagResNet
import torch
from signal_processing import filter_data
import random

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(0)
random.seed(0)
def get_sig(data):
    
    fData = filter_data(data[:12,:], highcut=50.0)
    
    return fData

def segment_sig(fData, n_segments):
    if fData.shape[1] < 3000:
        fData = np.pad(fData, pad_width=((0,0),(0,3000-fData.shape[1])), mode='constant', constant_values=0)
   
    segment_offset_len = (fData.shape[1] - 3000) //(n_segments+1)
    j_sigs = [segment_offset_len * k for k in range(n_segments)]

    # instances in a bag
    fDatas = np.array([[fData[:12,j_sig:j_sig+3000] for j_sig in j_sigs]])
    sample =torch.from_numpy(fDatas).type(torch.FloatTensor)
    return sample

def run_12ECG_classifier(data,header_data,loaded_model):

    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model['model']
    classes = loaded_model['classes']
    fData = get_sig(data)
    with torch.no_grad():
        model.eval()
        current_score = None
        sig_tensor = segment_sig(fData, n_segments)
        #print(sig_tensor.shape)
        outputs = model(sig_tensor.to(device))
        current_score = torch.sigmoid(outputs).cpu().numpy()[0] 
        current_label = np.round(current_score).astype(int)
        
        return current_label, current_score, classes

def load_trained_model(model_saved_path):
    model = ECGBagResNet(12, len(labels), n_segments).to(device)
    # load saved model
    model.load_state_dict(torch.load(model_saved_path+'/{}_model_final_40.dict'.format(run_name)))#, map_location=torch.device('cpu')))
    return model

def load_12ECG_model(input_directory):
    # load the model from disk 
    loaded_model = {}
    loaded_model['model'] = load_trained_model(input_directory)
    loaded_model['classes'] = [str(l) for l in labels]
    return loaded_model
