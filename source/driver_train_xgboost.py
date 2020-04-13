#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
#from evaluate_12ECG_score import compute_beta_score, compute_auc
import pandas as pd
from global_vars import headers, labels
    
def load_challenge_data(filename):


    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

  
# Find unique number of classes  
def get_classes(input_directory,files):

    classes=set()
    for f in files:
        g = f.replace('.mat','.hea')
        input_file = os.path.join(input_directory,g)
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)

def collect_data(input_directory, input_files):
    print('Extracting 12ECG features...')
    # Extract features
    num_files = len(input_files)
    datas = []
    header_datas = []
    #features = []
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory,f)
        data,header_data = load_challenge_data(tmp_input_file)
        datas.append(data)
        header_datas.append(header_data)
    return datas, header_datas

import pickle

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
                                                  highcut=30.0, fs=1000.0,
                                                  order=4)
            Data_filtered[chn, :] = filtered_ecg
        else:
            filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=0.5,
                                                  highcut=highcut, fs=1000.0,
                                                  order=4)
            Data_filtered[chn, :] = filtered_ecg            

    return Data_filtered

def findpeaks(data, spacing=1, limit=None):
    """Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param data: values
    :param spacing: minimum spacing to the next peak (should be 1 or more)
    :param limit: peaks should have value greater or equal
    :return:
    """
    len = data.size
    x = np.zeros(len+2*spacing)
    x[:spacing] = data[0]-1.e-6
    x[-spacing:] = data[-1]-1.e-6
    x[spacing:spacing+len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start : start + len]  # before
        start = spacing
        h_c = x[start : start + len]  # central
        start = spacing + s + 1
        h_a = x[start : start + len]  # after
        peak_candidate = np.logical_and(peak_candidate,
                                        np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind

from scipy.signal import find_peaks
def sep_rr_interval(bspm, percentile, distance=400, period_len=800, plot=False, p='', idx=''):
    # bspm: 252 x t
    
    rr_intervals = []
    total_time = bspm.shape[1]
    lead_vars = np.zeros((total_time,))
    for t in range(total_time):
        lead_vars[t] = np.var(bspm[:,t])
    lead_vars/=np.max(lead_vars)
    
    start_point = int(period_len/2)
    n_periods = int(len(lead_vars)/start_point)
    peak_values = []
    for i in range(n_periods-1):
        
        peak_value_ints = find_peaks(lead_vars[start_point*i:start_point*i+period_len])[0]
        if len(peak_value_ints) > 0:
            peak_value_ints_max = np.max(lead_vars[start_point*i:][peak_value_ints])
            peak_values.append(peak_value_ints_max)
    
    #print(peak_values)
    peak_value = 0
    if len(peak_values) > 0:
        peak_value = np.min(peak_values)
        
    peaks = find_peaks(lead_vars, height=peak_value, distance=distance)[0]
    if plot:
        plt.plot(lead_vars)
    for i in range(len(peaks)-1):
        if plot:
            plt.axvline(peaks[i], c='C1')
        rr_intervals.append((peaks[i], peaks[i+1]))
    if plot:
        if len(peaks) > 0:
            plt.axvline(peaks[-1], c='C1')
        intervals = np.diff(peaks)
        mean_intervals = np.mean(intervals)
        std_intervals = np.std(intervals)
        plt.axhline(peak_value, c='C2', label=str(peak_value))
        plt.legend()
        plt.title(str(len(intervals))+' ' + str(mean_intervals) + '_' + str(std_intervals))
        plt.savefig('../result/pAF/rr/'+str(idx) + '_'+str(p))
        
        plt.close()
    return rr_intervals, mean_intervals, std_intervals

        
def get_basic_info(header_data, labels):
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    age = None
    sex = None
    label = None
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            Dx = iline.split(': ')[1].split(',')[0]
            label = [ll in Dx for ll in labels]
    
    return [ptID, sex, age] + label
    

if __name__ == '__main__':
    
    print('Load data...')
    
    if not 'datas' in locals() or datas is None:
        datas = None
        header_datas = None
        classes = None
        with open('../saved/datas.pkl', 'rb') as f:
            datas = pickle.load(f)
        with open('../saved/header_datas.pkl', 'rb') as f:
            header_datas = pickle.load(f)
        with open('../saved/classes.pkl', 'rb') as f:
            classes = pickle.load(f)
            
#%%

    
#     df_X = df.drop(['Dx']+labels, axis=1).astype(float)
#     df_y = df[labels]
# #%%
#     # Train/Load model.
#     print('Training 12ECG model...')
#     params = {'max_depth': 500}
#     output_name = 'multiouput_xgb_fbeta2'
#     model = train_xgboost(df_X.to_numpy(), df_y.to_numpy(), output_name, params) 

#     # Testing
#     print('Testing ...')
#     preds_all = model.predict(df_X.to_numpy())
#     probs_all = model.predict_proba(df_X.to_numpy())
    
#     # Evaluate
#     print('Evaluate ...')
#     accuracy,f_measure,f_beta,g_beta = compute_beta_score(labels=df[labels].to_numpy(), 
#                        output=preds_all, 
#                        beta=2, num_classes=9)
    
#     auroc, auprc = compute_auc(labels=df[labels].to_numpy(), 
#                                probabilities=probs_all,
#                                num_classes=9)
    
#     output_string = 'AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure\n{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}'.format(auroc,auprc,accuracy,f_measure,f_beta,g_beta)
#     print(output_string)     
#     with open('xgb/saved/'+output_name+'score.txt', 'w') as f:
#         f.write(output_string)


    #%% 
    # classes=get_classes(input_directory,input_files)
    # datas, header_datas =collect_data(input_directory,input_files)
    # with open('datas.pkl', 'wb') as f:
    #     pickle.dump(datas, f)
    # with open('header_datas.pkl', 'wb') as f:
    #     pickle.dump(header_datas, f)
    # with open('classes.pkl', 'wb') as f:
    #     pickle.dump(classes, f)
        
    #%%
    # preds_all = []
    # probs_all = []
    # for ll in labels:
    #     preds = model[ll].predict(df_X)
    #     probs = model[ll].predict_proba(df_X)[:,1]
    #     preds_all.append(preds)
    #     probs_all.append(probs)
    #preds_all = np.array(preds_all).transpose()
    #probs_all = np.array(probs_all).transpose()
        
    
    #%%
    
    #%%


    #%%
    import matplotlib.pyplot as plt
    #fDatas = []
    #infos =[]
    period_len = 500
    rr_max = 650
    std_th = 200
    rr_min = 150
    for idx in range(480, len(datas)):
        # basic info
        info = get_basic_info(header_datas[idx], labels)
        
        
        # get data
        ptID = info[0]
        
        
        fData = filter_data(datas[idx][:,2000:], highcut=30.0)
        #fData = datas[idx][:,1000:]
        intervals, mean_intervals, std_intervals = sep_rr_interval(fData[6:], percentile=90, distance=rr_min, 
                                    plot=True, idx=idx, p=ptID, period_len=period_len)
        if len(intervals) == 0 or mean_intervals > rr_max or std_intervals > std_th:
            intervals, mean_intervals, std_intervals = sep_rr_interval(fData[:6], percentile=90, distance=rr_min, 
                                    plot=True, idx=idx, p=ptID, period_len=period_len)
        if len(intervals) == 0 or mean_intervals > rr_max or std_intervals > std_th:
            intervals, mean_intervals, std_intervals = sep_rr_interval(fData[:3], percentile=90, distance=rr_min, 
                                    plot=True, idx=idx, p=ptID, period_len=period_len)  
        info += [mean_intervals, std_intervals]
        print(str(idx) + ' ' + ptID + " {:.2f} {:.2f}".format(mean_intervals, std_intervals))
        
        if len(intervals) == 0 or mean_intervals > rr_max or std_intervals > std_th:
            break
        
        
        
        for i in range(len(intervals)):
            l, r = intervals[i]
            fDatas.append(fData[:,l:r])
            infos.append(info + [r-l]) # add interval length
            
        #     plt.plot(fData[0,l-100:r+100])
        #     plt.savefig('../result/pAF/seg/'+str(idx)+'_'+ptID+'_'+str(i))
        #     plt.close()
    #%%
    
    #%%
    # from sklearn.decomposition import FastICA
    # transform = FastICA(n_components=4)
    # fDataICA
    # = transform.fit_transform(fData.transpose()).transpose()
    #%%
    df_infos = pd.DataFrame(infos, columns=['ptID','Sex','Age']+labels)
    df_infos.to_csv('../saved/infos.csv')
    
#%%
    # lens = [fData.shape[1] for fData in fDatas]
    # print(lens)
    # [plt.plot(fData[6]) for fData in fDatas]

    #%%
    fDataRes=np.array([signal.resample(fData, 250, axis=1) for fData in fDatas])
    np.save('../saved/fDataRes.npy', fDataRes)
    #[plt.plot(fDataRe[6]) for fDataRe in fDataRes]
    
    #%%
    
    
    # #%%
    # from sklearn.preprocessing import normalize
    # #fDataReNorms=[fDataRe-np.tile(fDataRe[:,0],(250,1)).transpose() for fDataRe in fDataRes]
    # fDataReNorms=[normalize(fDataRe) for fDataRe in fDataRes]
    
    # [plt.plot(fDataReNorm[6]) for fDataReNorm in fDataReNorms]
    
    # #%%
    # df_X = np.array(fDataReNorms)
    
    # #%%
    # af_idx = np.argwhere(df_infos['RBBB']).flatten()
    # #%%
    # df_X_af = df_X[af_idx]
            
    # #%%
    # for X_af in df_X_af[::10]:
    #     plt.plot(X_af[0])
    #     plt.show()
    
    # #%%
    # from collections import Counter
    # print(Counter(np.sum(df_infos[labels], axis=1)))
    
    
    # #%%
    
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    # import torch.optim as optim
    
    # class SimpleNet(nn.Module) :
    # def __init__(self, n_input, n_h1) :
    #     "Defines the parameters of the model."
    #     super(TwoFullNet, self).__init__()
    #     self.fc1        = nn.Linear(n_input, n_h1)
    #     self.fc2        = nn.Linear(n_h1, n_h2)

    # def forward(self, x) :
    #     """
    #     Apply the model to some input data x.
    #     You can think of x as an image of size 28x28, but it is
    #     actually an Mx28x28 tensor, where M is the size of the
    #     mini-batch.
    #     """
    #     x = self.fc1( x )     
    #     x = F.relu(   x )     
    #     x = self.fc2( x )     
    #     return F.log_softmax(x) 
    
    # sn = SimpleNet(250, 250).forward(X)
    
    # #%%
    # def get_accuracy(output, y):
    
    #     output1 = np.argmax(output, axis=1)
    #     return balanced_accuracy_score(y, output1)
    
    # def get_auc(output, y):
    #     auroc, auprc = compute_auc(labels=y, 
    #                            probabilities=output,
    #                            num_classes=9)
        
    #     return auroc, auprc
        
    # #%%
    
    # from snippets.pytorchtools import EarlyStopping
    # from sklearn.model_selection import KFold
    
    # st = time.time()
    # patience = 200

    # for split in kf.split(X, y):
    #     model = SimpleNet(X).to(device)
        
    #     learning_rate = 0.01
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #     criterion = nn.CrossEntropyLoss(weight=weight)

    #     losses_train = []
    #     losses_test = []
    
    #     avg_losses_train = []
    #     avg_losses_test = []
    

    #     early_stopping = EarlyStopping(patience, verbose=False, 
    #                                saved_dir=saved_dir, save_name=str(i))
        
    #     for epoch in range(1000):
    #         train_loss = 0
    #         validation_loss = 0
            
    #         correct = 0
            
    #         optimizer.zero_grad()
    #         output_train = model(X[train_idx])
    #         output_test = model(X[test_idx])
            
    
    #         loss_train = criterion(output_train, y[train_idx])
    #         loss_test = criterion(output_test, y[test_idx])
            
    #         optimizer.zero_grad()
    #         loss_train.backward()
    #         optimizer.step()
            
    #         losses_train.append(loss_train.item())
    #         losses_test.append(loss_test.item())
            
    #         avg_loss_train = np.average(losses_train)
    #         avg_loss_test = np.average(losses_test)
            
    #         avg_losses_train.append(avg_loss_train)
    #         avg_losses_test.append(avg_loss_test)
            
    #         if epoch % 50 == 0:
    #             train_acc1 = get_accuracy((output_train.data).cpu().numpy(), 
    #                                                   (y[train_idx].data).cpu().numpy())
    #             test_acc1 = get_accuracy((output_test.data).cpu().numpy(), 
    #                                                  (y[test_idx].data).cpu().numpy())
                
    #             print('S{}/{} Train Loss: {:.6f}, Acc: {:.4f}| Validation Loss: {:.6f}, Acc: {:.4f}| {:.2f} min'.format(i, epoch, avg_loss_train, train_acc1, avg_loss_test, test_acc1, (time.time()-st)/60 ))
    # #            torch.save(model.state_dict(), saved_dir + '/saved_model/fold{}_{}'.format(i, epoch))
    
    #         early_stopping(avg_loss_test, model)
            
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break
        
    
