#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
import torch
from torchvision import transforms
from source.myeval import agg_y_preds
import matplotlib.pyplot as plt
import torchvision, torch
from PIL import Image

from torch import nn
from torchvision import models
    
class MultiCWTNet(nn.Module):
    def __init__(self, device, verbose=False):
        super(MultiCWTNet, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = self.increase_channels(self.resnet.conv1, num_channels=12, copy_weights=0)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 9)

        self.verbose = verbose
        
    def forward(self, xs):
        x = self.resnet(xs)
        return x
    
    
    def increase_channels(self, m, num_channels=None, copy_weights=0):
        """
        https://github.com/akashpalrecha/Resnet-multichannel/blob/master/multichannel_resnet.py
        
        takes as input a Conv2d layer and returns the a Conv2d layer with `num_channels` input channels
        and all the previous weights copied into the new layer.
        
        copy_weights (int): copy the weights of the channel (int)
        """
        # number of input channels the new module should have
        new_in_channels = num_channels if num_channels is not None else m.in_channels + 1
        
        # Creating new Conv2d layer
        new_m = nn.Conv2d(in_channels=new_in_channels, 
                          out_channels=m.out_channels, 
                          kernel_size=m.kernel_size, 
                          stride=m.stride, 
                          padding=m.padding,
                          bias=False)
        
        # Copying the weights from the old to the new layer
        new_m.weight[:, :m.in_channels, :, :] = m.weight.clone()
        
        #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - m.in_channels): # 12 - 3
            channel = m.in_channels + i # 3，4，5，6，7，8，9，10，11
            new_m.weight[:, channel:channel+1, :, :] = m.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_m.weight = nn.Parameter(new_m.weight)

        return new_m
    
import pywt
def cwt(signals, width):

    num_steps = signals.shape[1]
    x = np.arange(num_steps) * 0.002
    delta_t = x[1] - x[0] # 500Hz 0.02s

    wavelet_type = 'morl'
    scales = np.linspace(8,200,width)
    
    coef_norms = []
    for chn in list(range(12)):
        coef, freqs = pywt.cwt(signals[chn], scales, wavelet_type, delta_t)
        coef_norm = (coef-np.min(coef))/(np.max(coef) - np.min(coef))
        coef_norms.append(coef_norm)
        
    return np.array(coef_norms)


from scipy import signal 

def butter_bandpass(lowcut, highcut, fs, order=5, vis=False):
    nyq = 0.5 * fs # fs / 2
    low = lowcut / nyq # lowcut * 2 / fs
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    # visualize the filter
    if vis:
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order) # fs / (2 * pi)
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(low, color='green') # cutoff frequency
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # lowcut, fs in Hz
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    
    return y

def filter_data(Data, highcut=None):
    Data_filtered = np.zeros(Data.shape)
    chns = Data.shape[0]
    for chn in range(chns):
        if highcut is None:
            filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=0.5,
                                                  highcut=30.0, fs=500.0,
                                                  order=4)
            Data_filtered[chn, :] = filtered_ecg
        else:
            filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=0.5,
                                                  highcut=highcut, fs=500.0,
                                                  order=4)
            Data_filtered[chn, :] = filtered_ecg            

    return Data_filtered

def get_image(data):
    
    n_segments = int(data.shape[1]/3000)
    resize = torchvision.transforms.Resize((224, n_segments*224))
    fData = filter_data(data[:,:n_segments*3000], highcut=50.0)
    
    coef = cwt(fData, width=224) 
    coef = coef.transpose((1,2,0))
    
    data_img0 = Image.fromarray((coef[:,:,:3] * 255).astype(np.uint8)) 
    data_img1 = Image.fromarray((coef[:,:,3:6] * 255).astype(np.uint8)) 
    data_img2 = Image.fromarray((coef[:,:,6:9] * 255).astype(np.uint8)) 
    data_img3 = Image.fromarray((coef[:,:,9:12] * 255).astype(np.uint8)) 
    data_imgs = [ resize(data_img0), 
                  resize(data_img1), 
                  resize(data_img2), 
                  resize(data_img3)]
    
    n_segments = 5
    shift_len = int((data_imgs[0].size[0]-224) / (n_segments-1))
    imgs = []
    for i in range(n_segments):
        # each channel of each segment
        img_chns = []
        for chn in range(4):
            img = data_imgs[chn].crop((i*shift_len,0,i*shift_len+224,224)) # 0, 0, 224, 224 left, upper, right, and lower
            img_chns.append(img)
        imgs.append(img_chns)
    
    return imgs


transform =  transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])

def run_12ECG_classifier(data,header_data,classes,model):
    data_imgs = get_image(data)
    n_segments = 5
    with torch.no_grad():
        model.eval()
        imgs_tensors = []
        for idx in range(n_segments):
            images = [data_imgs[idx][chn] for chn in range(4)]
            imgs_tensor = torch.cat([transform(data_img) for data_img in images],0)
            imgs_tensors.append(imgs_tensor)
        imgs_tensors = torch.stack(imgs_tensors)
        output = model(imgs_tensors)
        y_prob_tensor, _ = agg_y_preds(output)
        current_score = y_prob_tensor.data.numpy()
        current_label = np.round(current_score).astype(int)

        return current_label, current_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

def load_cwt_nn_model(model_saved_path):
    device = torch.device('cpu')
    model = MultiCWTNet(device, verbose=False)
    # load saved model
    model.load_state_dict(torch.load(model_saved_path, map_location=device))
    model.eval()
    return model

def load_12ECG_model():
    # load the model from disk 
    model_saved_path = 'saved/modelMultiCWTFull/MutliCWTNetFull0_model.dict' 
    model = load_cwt_nn_model(model_saved_path)
    return model
