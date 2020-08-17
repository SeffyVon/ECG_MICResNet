#!/usr/bin/env python

import numpy as np, os, sys
from global_vars import labels
from get_12ECG_features import get_12ECG_features
from MultiCWTNet import MultiCWTNet
from ImageMultichannelDataset import img_transforms
import torch
import torchvision
from PIL import Image
from make_cwt import filter_data, cwt

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def get_image(data):
    
    n_segments = int(data.shape[1]/3000)
    resize = torchvision.transforms.Resize((224, n_segments*224))
    fData = filter_data(data[:,:n_segments*3000], highcut=50.0)
    
    coef = cwt(fData, width=40) 
    coef = coef.transpose((1,2,0))
    
    data_img0 = Image.fromarray((coef[:,:,:3] * 255).astype(np.uint8)) 
    data_img1 = Image.fromarray((coef[:,:,3:6] * 255).astype(np.uint8)) 
    data_img2 = Image.fromarray((coef[:,:,6:9] * 255).astype(np.uint8)) 
    data_img3 = Image.fromarray((coef[:,:,9:12] * 255).astype(np.uint8)) 
    data_imgs = [resize(data_img0), 
                  resize(data_img1), 
                  resize(data_img2), 
                  resize(data_img3)]
    
    return data_imgs

def run_12ECG_classifier(data,header_data,loaded_model):

    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model['model']
    classes = loaded_model['classes']
    data_imgs = get_image(data)

    with torch.no_grad():
        model.eval()
        imgs_tensor = img_transforms['test'](data_imgs).reshape((1,12,224,224))
        outputs = model(imgs_tensor.to(device))
        current_score = torch.sigmoid(outputs).cpu().numpy()[0]
        current_label = np.round(current_score)
    
        return current_label, current_score, classes

def load_trained_model(model_saved_path):
    model = MultiCWTNet(len(labels), verbose=False).to(device)
    # load saved model
    model.load_state_dict(torch.load(model_saved_path+'/modelMultiCWTFull_test_model.dict'))#, map_location=torch.device('cpu')))
    return model

def load_12ECG_model(input_directory):
    # load the model from disk 
    loaded_model = {}
    loaded_model['model'] = load_trained_model(input_directory)
    loaded_model['classes'] = [str(l) for l in labels]
    return loaded_model
