#!/usr/bin/env python

import numpy as np, os, sys
from global_vars import labels
from get_12ECG_features import get_12ECG_features
from MultiCWTNet import MultiCWTNet

def run_12ECG_classifier(data,header_data,loaded_model):


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model['model']
    classes = loaded_model['classes']

    current_score = model(data)
    current_label = np.round(current_score)
    
    return current_label, current_score, classes

def load_trained_model(model_saved_path):
    model = MultiCWTNet(len(labels), verbose=False)
    
    # load saved model
    model.load_state_dict(torch.load(model_saved_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_12ECG_model(input_directory):
    # load the model from disk 
    loaded_model = {}
    loaded_model['model'] = load_trained_model(input_directory)
    loaded_model['class'] = labels
    return loaded_model
