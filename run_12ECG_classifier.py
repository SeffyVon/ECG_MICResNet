#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from source.driver_train_xgboost import filter_data, sep_rr_interval, get_basic_info
from source.dnn.SimpleNet import SimpleNet
import torch
from scipy import signal 
from sklearn.preprocessing import normalize
def run_12ECG_classifier(data,header_data,classes,model):
    # Use your classifier here to obtain a label and score for each class. 
    
    fDatas = []
    #infos =[]
    fData = filter_data(data[:,1000:], highcut=50.0)
    #fData = datas[idx][:,1000:]
    intervals = sep_rr_interval(fData[:3], height=0.2, distance=200, plot=False)

    # basic info
    #info = get_basic_info(header_datas[idx], labels)
    
    # get data
    #ptID = info[0]
    #print(str(idx) + ' ' + ptID)
    for i in range(len(intervals)):
        l, r = intervals[i]
        fDatas.append(fData[:,l:r])
        #infos.append(info)
            
    fDataRes=np.array([signal.resample(fData, 250, axis=1) for fData in fDatas])
    fDataReNorms=np.array([normalize(fDataRe) for fDataRe in fDataRes])
    
    X = torch.FloatTensor(fDataReNorms)
    
    try:
        output = model(X)
        current_score = np.mean(torch.sigmoid(output).data.numpy(), axis=0)
        current_label = np.rint(current_score).astype(int)
   # print(current_score.shape, current_label.shape)
    except:
        print(fDataRes, header_data)
        current_label = np.zeros((9,), dtype=int)
        current_score = np.zeros((9,))
    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    
    model = SimpleNet(250, 250, 10, 10)
    model_saved_path = 'saved/model/SimpleNet0_model.dict'
    cpu_device = torch.device('cpu')
    model.load_state_dict(torch.load(model_saved_path, map_location=cpu_device))
    model.eval()
    return model
