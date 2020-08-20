import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, sys
from manipulations import get_classes, get_classes_from_header, get_Fs_from_header, load_challenge_data
from get_12ECG_features import get_12ECG_features
from saved_data_io import read_file, write_file
from global_vars import disable_tqdm
import pywt
from scipy import signal 

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs # fs / 2
    low = lowcut / nyq # lowcut * 2 / fs
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    # lowcut, fs in Hz
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.filtfilt(b, a, data)
    
    return y

def filter_data(Data, highcut):
    Data_filtered = np.zeros(Data.shape)
    chns = Data.shape[0]
    for chn in range(chns):
        filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=0.5,
                                              highcut=highcut, fs=500.0,
                                              order=4)
        Data_filtered[chn, :] = filtered_ecg            

    return Data_filtered



def write_signal(recordings_datasets, headers_datasets, output_directory, disable_tqdm=disable_tqdm):

    if not os.path.isdir(output_directory + '/sig'):
        os.mkdir(output_directory+ '/sig')

    if not os.path.isdir(output_directory + '/features'):
        os.mkdir(output_directory+ '/features')

    datasets = np.sort(list(headers_datasets.keys()))
    features = []
    for dataset in datasets:

        print('Dataset ', dataset)

        # compute CWT every #max_num_cwt(1000) recordings
        # 0-1000, 1000-2000, 2000-3000, ..., num(input_files)
        recordings = recordings_datasets[dataset]
        headers = headers_datasets[dataset]
        num_files = len(recordings)
        print("#recordings: ", num_files)
        for i in tqdm(range(num_files), leave=False, disable=disable_tqdm):
            header = headers[i]
            filename = header[0].split(' ')[0].split('.')[0]
            sig_file = output_directory + '/sig/' + filename + '.npy'
            #feature_file = output_directory + '/features/' + filename + '.npy'

            # if os.path.exists(sig_file) and os.path.exists(feature_file):
            #     continue
            data = recordings[i]

            if not os.path.exists(sig_file):
                fData = None
                if np.sum(data) != 0:
                    fData = filter_data(data[:12,:], highcut=50.0)
                else:
                    fData = np.zeros((12, data.shape[1]), dtype=np.float64)

                write_file(sig_file, fData)

            #if not os.path.exists(feature_file):
            feature = get_12ECG_features(data, header)
            features.append(feature)
               # write_file(feature_file, features)
            
                    
        print('Done.')
        
    features = np.array(features)
    return features
