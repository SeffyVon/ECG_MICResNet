# -*- coding: utf-8 -*-
from get_12ECG_features import detect_peaks
from scipy import stats
import numpy as np

def extract_features_xgboost(datas, header_datas):
    features = []
    for data, header_data in zip(datas, header_datas):
        feature = extract_feature_xgboost(data, header_data)
        features.append(feature)
    return np.array(features)

def extract_feature_xgboost(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    if num_leads != 12:
        raise Exception
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age) if tmp_age != 'NaN' else 57
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex = 0
        elif iline.startswith('#Dx'):
            dx = iline.rstrip().split(': ')[1]

    
#   We are only using data from lead1
    def extract_lead_feature(lead_i):
        peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[lead_i])
       
    #   mean
        mean_RR = np.mean(idx/sample_Fs*1000)
        mean_Peaks = np.mean(peaks*gain_lead[lead_i])
    
    #   median
        median_RR = np.median(idx/sample_Fs*1000)
        median_Peaks = np.median(peaks*gain_lead[lead_i])
    
    #   standard deviation
        std_RR = np.std(idx/sample_Fs*1000)
        std_Peaks = np.std(peaks*gain_lead[lead_i])
    
    #   variance
        var_RR = stats.tvar(idx/sample_Fs*1000)
        var_Peaks = stats.tvar(peaks*gain_lead[lead_i])
    
    #   Skewness
        skew_RR = stats.skew(idx/sample_Fs*1000)
        skew_Peaks = stats.skew(peaks*gain_lead[lead_i])
    
    #   Kurtosis
        kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
        kurt_Peaks = stats.kurtosis(peaks*gain_lead[lead_i])
    
        lead_features = [mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,
                    std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,
                    kurt_Peaks]
        
        return lead_features


    features = [[age,sex,dx]]
    for i in range(num_leads):
        lead_i_features = extract_lead_feature(i)
        features.append(lead_i_features)
    
    features = np.concatenate(features)
    return features
