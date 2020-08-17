import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, torchvision 
from PIL import Image
import numpy as np
import os, sys
from manipulations import get_classes, get_classes_from_header, get_Fs_from_header, load_challenge_data
from saved_data_io import read_file, write_file
import pywt
from scipy import signal 

def cwt(signals,name='', width=10, wavelet_type = 'morl'):
    #print(signals.shape)
    # signals: channels x time
    num_steps = signals.shape[1]
    x = np.arange(num_steps) * 0.002
    delta_t = x[1] - x[0] # 500Hz 0.02s
    #print(delta_t)
    #Freq (5, 100)
    if wavelet_type == 'morl':
        scales = np.linspace(8,200,width)
    elif wavelet_type == 'mexh':
        scales = np.linspace(5,50,100)
    elif wavelet_type == 'gaus8':
        scales = np.linspace(12,120,100)
    else: # cmor
        scales = np.linspace(12,120,100)
    
    coef_norms = []
    for chn in list(range(12)):
        coef, freqs = pywt.cwt(signals[chn], scales, wavelet_type, delta_t)
        coef_norm = (coef-np.min(coef))/max(np.max(coef) - np.min(coef), 0.0001)
        coef_norms.append(coef_norm)
    return np.array(coef_norms)



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs # fs / 2
    low = lowcut / nyq # lowcut * 2 / fs
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # lowcut, fs in Hz
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    
    return y

def filter_data(Data, highcut=30.0):
    Data_filtered = np.zeros(Data.shape)
    chns = Data.shape[0]
    for chn in range(chns):
        filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=0.5,
                                              highcut=highcut, fs=500.0,
                                              order=4)
        Data_filtered[chn, :] = filtered_ecg            

    return Data_filtered



def make_cwt(recordings_datasets, output_directory):
    
    max_num_cwt = 1000
    for dataset in recordings_datasets.keys():

        print('Dataset ', dataset)


        # compute CWT every #max_num_cwt(1000) recordings
        # 0-1000, 1000-2000, 2000-3000, ..., num(input_files)
        recordings = recordings_datasets[dataset]
        num_files = len(recordings) 
        print("#recordings: ", num_files)
        K = num_files // max_num_cwt
        for k in tqdm(range(K+1), leave=False):
            data_imgs = []
            for i in tqdm(range(k*max_num_cwt, (k+1)*max_num_cwt), leave=False):
                if i < num_files:
                    data = recordings[i]
                    n_segments = max(1,min(data.shape[1]//3000,11))
                    resize = torchvision.transforms.Resize((224, n_segments*224))
                    
                    fData = np.zeros((data.shape[0], n_segments*3000))
                    if np.sum(data) != 0:
                        fData = filter_data(data[:,:n_segments*3000], highcut=50.0)

                    coef = cwt(fData, width=40) 
                    coef = coef.transpose((1,2,0))
                    data_img0 = Image.fromarray((coef[:,:,:3] * 255).astype(np.uint8)) 
                    data_img1 = Image.fromarray((coef[:,:,3:6] * 255).astype(np.uint8)) 
                    data_img2 = Image.fromarray((coef[:,:,6:9] * 255).astype(np.uint8)) 
                    data_img3 = Image.fromarray((coef[:,:,9:12] * 255).astype(np.uint8)) 
                    data_imgs.append([resize(data_img0), 
                                      resize(data_img1), 
                                      resize(data_img2), 
                                      resize(data_img3)])
            write_file(output_directory + '/data_imgs_dataset{}_{}.pkl'.format(dataset, k), 
                       data_imgs)

        # aggregate CWTs
        data_imgs = []
        for k in range(K+1):
            data_imgs += read_file(output_directory + '/data_imgs_dataset{}_{}.pkl'.format(dataset, k))
        
        # saved
        write_file(output_directory + '/data_imgs_dataset{}.pkl'.format(dataset), 
                   data_imgs)

        del data_imgs
        
        print('Done.')
