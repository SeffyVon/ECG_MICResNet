"""
Filtering
"""
from scipy import signal 
import numpy as np
import matplotlib.pyplot as plt

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

def filter_data(Data, fs, highcut):
    Data_filtered = None
    if Data.ndim == 1:
        Data_filtered = butter_bandpass_filter(Data, lowcut=2,
                                                  highcut=highcut, fs=fs,
                                                  order=4)
    else:
        Data_filtered = np.zeros(Data.shape)
        chns = Data.shape[0]
        for chn in range(chns):
            filtered_ecg = butter_bandpass_filter(Data[chn,:], lowcut=2,
                                                  highcut=highcut, fs=fs,
                                                  order=4)
            Data_filtered[chn, :] = filtered_ecg


    return Data_filtered

def despike(signal, window=20):
    threshold=np.std(signal)*4
    signal_copy = np.copy(signal)
    for i in range(len(signal)-window):
        if np.max(signal_copy[i:i+window]) - np.min(signal_copy[i:i+window]) > threshold:
            #print("yes")
            #signal_copy[i:i+window] = np.linspace(signal[i], signal[i+window], window)
            signal_copy[i:i+window] = im.median_filter(signal_copy[i:i+window], len(signal_copy[i:i+window]))   
    return signal_copy

def myfilter(raw, fs, vis=False):
    ans0 = filter_data(raw, highcut=30.0, fs=fs) 
    if vis:
        plt.plot(ans0)
        plt.title('highpass')
        plt.show()
   
    return ans0

"""
Pan Tompkin
"""

def derivative(x):
    # derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    # Make impulse response
    h = np.array([-1, -2, 0, 2, 1])/8.0
    y = signal.convolve(x ,h)
    y = y[2:]
    y = y/max(abs(y))
    return y

def squaring(x):
    y = x**2
    y = y/max(abs(y))
    return y


def MA(x):
    # Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    h = np.ones(31)/31
    y = signal.convolve(x, h)
    y = y[15:]
    y = y/max(abs(y))
    return y

from sklearn.decomposition import PCA
def Pan_Tompkins_QRS(ecg, verbose=False):
    """
    Use Pan Tompkins algorithm to extract QRS
    cannot cope with negative R yet!
    :param ecg:
    :param verbose:
    :return:
    """
    filtered_ecg = ecg

    filtered_ecg = filtered_ecg/max(abs(filtered_ecg))
    
    differentiated_ecg = derivative(filtered_ecg)
    squaring_ecg = squaring(differentiated_ecg)
    integrated_ecg = MA(squaring_ecg)
    max_h = max(integrated_ecg)
    thresh = max(np.mean(integrated_ecg)*max_h, 0.1)
    if verbose:
        print("threshold:{}".format(thresh))
    poss_reg = (integrated_ecg>thresh)
    poss_reg=poss_reg.astype(int)
    
    # the left points of the integrated signal above the threshold
    left = np.where(np.diff(np.insert(poss_reg, 0, 0))==1)[0]
    left=left.astype(int)
    #left=left-1 # give it a little bit more offset

    # the right points of the integrated signal below the threshold
    right = np.where(np.diff(np.insert(poss_reg, -1, 0))==-1)[0]
    right=right.astype(int)

    if verbose:
        print("left:", left, "right:", right)
#     if len(left) > 2:
#        left = left[[0,2]]
#        right = right[[0,2]]

    if left[0] < 100:
        left = left[1:]
        right = right[1:]
        
    if left[-1] > len(filtered_ecg) - 100:
        left = left[:-1]
        right = right[:-1]
    
    # scan from left to right
    R_loc=np.zeros(len(left), dtype=int)
    Q_loc=np.zeros(len(left), dtype=int)
    S_loc=np.zeros(len(left), dtype=int)
            
    for i in range(len(left)):
        R_loc[i] = np.argmax(filtered_ecg[left[i]:right[i]+1])
        R_loc[i] = R_loc[i]+left[i]

        # R_left
        R_left = R_loc[i]
        while R_left-1 and filtered_ecg[R_left-1] > filtered_ecg[R_left]:
            R_left = R_left-1
        if filtered_ecg[R_left] > filtered_ecg[R_loc[i]]:
            R_loc[i] = R_left

        # R_right
        R_right = R_loc[i]            
        while R_right+1 < right[i] and filtered_ecg[R_right+1] > filtered_ecg[R_right]:
            R_right = R_right+1
        if filtered_ecg[R_right] > filtered_ecg[R_loc[i]]:
            R_loc[i] = R_right

        left[i] = min(left[i], R_loc[i])
        right[i] = max(right[i], R_loc[i])
        
        # Adjusting Q_loc
        Q_loc[i] = np.argmin(filtered_ecg[left[i]:R_loc[i]+1])
        Q_loc[i] = Q_loc[i]+left[i]   
        while Q_loc[i]-1 and filtered_ecg[Q_loc[i]-1] < filtered_ecg[Q_loc[i]]:
            Q_loc[i] = Q_loc[i] - 1

        # Adjusting S_loc
        S_loc[i] = np.argmin(filtered_ecg[R_loc[i]:right[i]+1])
        S_loc[i] = S_loc[i]+R_loc[i]+1
        while S_loc[i]+1 < len(ecg)-1 and filtered_ecg[S_loc[i]+1] < filtered_ecg[S_loc[i]]:
            S_loc[i] = S_loc[i]+1
            
    
    if verbose:
        print("Q locations:", Q_loc)
        print("R locations:", R_loc)
        print("S locations:", S_loc)
    
    return Q_loc, R_loc, S_loc, integrated_ecg, left, right, thresh

def remove_QRS(filtered_ecg, Q_loc, S_loc, verbose=False):
    """
    cancel QRS points to the Q value to facilitate the search of T peak
    """
    
    assert(len(Q_loc) == len(S_loc))
    ecg_no_QRS = np.array(filtered_ecg, copy=True)  
    first_drv = filtered_ecg - np.roll(filtered_ecg, 1)
    end = np.zeros(len(Q_loc), dtype=int)
    
    for i in range(len(Q_loc)):
        end[i] = S_loc[i]
        boundary = len(first_drv)
        if i+1 < len(Q_loc):
            boundary = Q_loc[i+1]
            
        while end[i] + 1 < boundary  and filtered_ecg[end[i]] < 0:
            end[i] = end[i]+1
        # end is the first point that first_drv < 0 after S_loc[i]
        if verbose:
            print("T start detection: first_drv:", first_drv[end[i]], "filtered_ecg:", filtered_ecg[end[i]])
        ecg_no_QRS[Q_loc[i]:end[i]] = filtered_ecg[Q_loc[i]]
        
    return ecg_no_QRS, end

from scipy.signal import find_peaks

def find_AD(ecg_12leads, vis=False):
    """
    LAD = False
    RAD = False
    if R_V1 > 0 and R_aVF > 0:
        pass
    elif R_V1 > 0 and R_aVF < 0:
        LAD = True
    elif R_V1 < 0 and R_aVF > 0:
        RAD = True
    """
    diagnosis = "Normal"
    peak_indices = {}

    for i in [0,1,5]:    
        peak_indices[i],_ = find_peaks(np.abs(ecg_12leads[i,:]), height=np.percentile(np.abs(ecg_12leads[i,:]), 99))
        if vis:
            plt.plot(ecg_12leads[i,:], c='C{}'.format(i))
            plt.scatter(peak_indices[i], ecg_12leads[i,peak_indices[i]], c='C{}'.format(i))
            plt.show()
            plt.close()

    pos_R_I = True if np.sum(ecg_12leads[0,peak_indices[0]] > 0) > np.sum(ecg_12leads[0,peak_indices[0]] < 0) else False
    pos_R_II = True if np.sum(ecg_12leads[0,peak_indices[1]] > 0) > np.sum(ecg_12leads[0,peak_indices[1]] < 0) else False
    pos_R_aVF = True if np.sum(ecg_12leads[5,peak_indices[5]] > 0) > np.sum(ecg_12leads[5,peak_indices[5]] < 0) else False

    if not pos_R_I and pos_R_aVF :
        diagnosis = 'RAD'
    elif pos_R_I and (not pos_R_aVF or not pos_R_II):
        diagnosis = 'LAD'
    return diagnosis

from itertools import groupby
from operator import itemgetter

def extract_QRST(ecg_12leads, vis=False, vis_res=False, only_qrs=False, verbose=False,
                 filter_twice=False, qrs_chn=12, code=None, title=None):
    """
    To extract the QRST.
    Cannot cope with the negative R yet!
    
    """

    filtered_ecg = None
    filtered_ecg2 = None
    if qrs_chn == 12:
        # frontal leads
        filtered_ecg = PCA(1).fit_transform(ecg_12leads[:3,:].transpose()).flatten() 
        peak_indices,_ = find_peaks(np.abs(filtered_ecg), height=np.percentile(np.abs(filtered_ecg), 98))
        if vis:
            plt.plot(filtered_ecg)
            plt.scatter(peak_indices, filtered_ecg[peak_indices])
            plt.show()
            plt.close()

        sign = 1 if np.sum(filtered_ecg[peak_indices] > 0) > np.sum(filtered_ecg[peak_indices] < 0) else -1
        filtered_ecg *= sign
        
        
        # all leads
        filtered_ecg2 = None
#         if np.sum(ecg_12leads[[7,8],:]) != 0:
#             filtered_ecg2 = PCA(1).fit_transform(ecg_12leads[[7,8],:].transpose()).flatten() 
#         else:
#             
        filtered_ecg2 = PCA(1).fit_transform(ecg_12leads[[0,1],:].transpose()).flatten() 
            
#         peak_indices,_ = find_peaks(np.abs(filtered_ecg2), height=np.percentile(np.abs(filtered_ecg2), 98))
#         sign = 1 if np.sum(filtered_ecg2[peak_indices] > 0) > np.sum(filtered_ecg2[peak_indices] < 0) else -1
#         filtered_ecg2 *= sign
        peak_indices,_ = find_peaks(np.abs(ecg_12leads[0,:]), height=np.percentile(np.abs(ecg_12leads[0,:]), 99))
        pos_R_I = 1 if np.sum(ecg_12leads[0,peak_indices] > 0) > np.sum(ecg_12leads[0,peak_indices] < 0) else -1
        filtered_ecg2 = ecg_12leads[0,:] #* pos_R_I
        
    elif qrs_chn == 13:
        frontal_verticle_leads =  PCA(2).fit_transform(ecg_12leads[[0, 5],:].transpose()).transpose()
        filtered_ecg = frontal_verticle_leads[0,:] * pos_R_I * pos_R_aVF
    else:
        filtered_ecg = ecg_12leads[qrs_chn,:]

    if vis:    
        plt.figure(figsize=(15,3))
        plt.plot(frontal_leads.transpose())
        plt.plot(filtered_ecg, c='black')
        plt.title('signal for QRST segmentation')
        plt.show()
        plt.close()
        
    Q_loc, R_loc, S_loc, integrated_ecg, left, right, thresh = Pan_Tompkins_QRS(filtered_ecg, verbose=verbose)
    
    if vis:    
        plt.figure(figsize=(20,10))
        plt.hlines(thresh, 0, len(filtered_ecg))
        plt.plot(filtered_ecg, label='filtered_ecg')
        plt.plot(integrated_ecg, label='integrated signal')
        plt.scatter(left, integrated_ecg[left], label='left')
        plt.scatter(right, integrated_ecg[right], label='right')
        plt.scatter(Q_loc, filtered_ecg[Q_loc], label='Q', marker='^', s=100)
        plt.scatter(R_loc, filtered_ecg[R_loc], label='R', marker='^', s=100)
        plt.scatter(S_loc, filtered_ecg[S_loc], label='S', marker='^', s=100)
        plt.legend(loc='best')
        #plt.title("Detect QRS of patient {} phase {}".format(patient_num, phase_num))
        #plt.savefig('../result/pAF/seg/' + str(patient_num) + '_' + str(phase_num) + '_seg.png')
        plt.show()
        plt.close()
        
    if only_qrs:
        return Q_loc, R_loc, S_loc, None, None, None

    # Now we can calculate the variance of the channels to detect the T waves :)
    _, T_start_loc = remove_QRS(filtered_ecg, Q_loc, S_loc)

    if verbose:
        print ("Q_loc:", Q_loc,"R_loc:", R_loc, "S_loc:", S_loc)
    
    GAP_SINCE_S = 0 # 500 Hz, change 100 to 50
    
    T_peak_loc = np.zeros(len(T_start_loc)-1, dtype=int)
    T_end_loc = np.zeros(len(T_start_loc)-1, dtype=int)
    time_vars = np.zeros(ecg_12leads.shape[1])
    
    filtered_ecg2 = butter_bandpass_filter(filtered_ecg2, lowcut=0.5, highcut=10, fs=500, order=2)
    
    # my method
#     for i in range(len(T_start_loc)-1):
#         time_vars[T_start_loc[i]:Q_loc[i+1]] = filtered_ecg2[T_start_loc[i]:Q_loc[i+1]]
#         #from this T_start_loc to next Q
        
         
#         # T peak is defined within the start to the 2/3 of the S-Q segment
#         T_peak_bound = int(1/3 * T_start_loc[i] + 2/3 * Q_loc[i+1]) 
#         if T_peak_bound <= T_start_loc[i]:
#             T_end_loc[i] = T_start_loc[i]
#             continue
#         if verbose:
#             print("T_start_loc[i]", T_start_loc[i], "T_peak_bound", T_peak_bound)
            
#         T_peak_loc[i] = np.argmax(time_vars[T_start_loc[i]:T_peak_bound]) + T_start_loc[i]
#         if verbose:
#             print("T_peak_loc", T_peak_loc[i])

#         # backward search for the end of the T end
#         T_end_loc[i] = T_peak_loc[i]
#         while T_end_loc[i]<Q_loc[i+1] and (time_vars[T_end_loc[i]+1]<time_vars[T_end_loc[i]]):
#             T_end_loc[i] = T_end_loc[i] + 1
#         if verbose:
#             print("T_end_loc", T_end_loc[i])

    # new method

    intervals = np.zeros((len(T_start_loc)-1,), dtype=int)
    
    RR = np.median([R_loc[i+1]-R_loc[i] for i in range(len(T_start_loc)-1)])
    fs = 1./500
    W1 = int(35 * fs * RR )
    W2 = int(200 * fs * RR)
    RTmin = int(35 * fs * RR )
    RTmax = int(500 * fs * RR )

    if verbose:
        print("W1, W2, RTmin, RTmax", W1, W2, RTmin, RTmax)
        
    if vis:
        plt.figure(figsize=(20,5))
    
    # T wave inversion
    for i in range(len(T_start_loc)-1):
        time_vars[T_start_loc[i]:Q_loc[i+1]] = filtered_ecg2[T_start_loc[i]:Q_loc[i+1]]
    
    T_wave_inv = False    
#     arg_T_peak = np.argmax(np.abs(time_vars))
#     if time_vars[arg_T_peak] < 0:
#         T_wave_inv = True

    if T_wave_inv:
        filtered_ecg2 *= -1        
        
    for i in range(len(T_start_loc)-1):
        RR = R_loc[i+1]-R_loc[i]    
        time_vars[T_start_loc[i]:Q_loc[i+1]] = filtered_ecg2[T_start_loc[i]:Q_loc[i+1]]

        MA_peak = np.convolve(time_vars, np.ones((W1,))/W1, mode='same')
        MA_Twave = np.convolve(time_vars, np.ones((W2,))/W2, mode='same')        

        roi_indices = np.argwhere(MA_peak[T_start_loc[i]:Q_loc[i+1]]>MA_Twave[T_start_loc[i]:Q_loc[i+1]]).flatten() + T_start_loc[i]
        
        for k,g in groupby(enumerate(roi_indices), lambda value:value[0]-value[1]):
            grouping = list(map(itemgetter(1), g))
            if len(grouping)>1 and intervals[i] == 0 and grouping[-1] - grouping[0] > W1:
                if grouping[0] -  R_loc[i] > RTmin and grouping[-1]- R_loc[i] < RTmax:
                    intervals[i] = grouping[-1] - grouping[0]
                    T_start_loc[i] = grouping[0]
                    T_end_loc[i] = grouping[-1]
                    T_peak_loc[i] = np.argmax(time_vars[T_start_loc[i]: T_end_loc[i]])
                    if vis:
                        plt.axvspan(grouping[0], grouping[-1], alpha=0.4, color='C3')
                elif grouping[-1]- R_loc[i] > RTmax:
                    if vis:
                        plt.axvspan(grouping[0], grouping[-1], alpha=0.4, color='C4')                

    MA_peak = np.convolve(filtered_ecg2, np.ones((W1,))/W1, mode='same')
    MA_Twave = np.convolve(filtered_ecg2, np.ones((W2,))/W2, mode='same')
    
    if vis:
        plt.plot(filtered_ecg2, label='I, aVR post')
        plt.plot(MA_peak, label='MA_peak')
        plt.plot(MA_Twave, label='MA_Twave')
        plt.scatter(Q_loc, filtered_ecg2[Q_loc], label='Q', s=100)
        plt.scatter(R_loc, filtered_ecg2[R_loc], label='R', s=100)
        plt.scatter(S_loc, filtered_ecg2[S_loc], label='S', s=100)
        plt.legend()
        #plt.savefig('../explore/QRST/'+str(code))
        plt.show()    
        plt.close()
  
    return Q_loc, R_loc, S_loc, T_start_loc, T_peak_loc, T_end_loc, filtered_ecg, filtered_ecg2


"""
CWT
"""
import pywt
def cwt(signal,name='', wavelet_type = 'morl', vis=False):
    num_steps = len(signal)
    x = np.arange(num_steps) * 1e-3
    delta_t = x[1] - x[0]
    #Freq (5, 100)
    if wavelet_type == 'morl':
        scales = np.linspace(16,200,100)
    elif wavelet_type == 'mexh':
        scales = np.linspace(5,50,100)
    elif wavelet_type == 'gaus8':
        scales = np.linspace(12,120,100)
    else: # cmor
        scales = np.linspace(12,120,100)
        
    coefs, freqs = pywt.cwt(signal, scales, wavelet_type, delta_t)
    if vis:
        #plt.matshow(coefs.astype(float)) 
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        plt.imshow(coefs.astype(float), aspect='auto', extent=(1, len(signal), freqs[-1], freqs[0]), cmap='RdBu_r')
        plt.colorbar()
        

        #print((freqs[0], freqs[-1]))
        plt.yticks(freqs)
        plt.ylabel('Frequency(Hz)')
        plt.xlabel('Time(ms)')
        
        ax = ax.twinx()
        ax.plot(signal)
        #plt.savefig('../result/sim_bspms/cwt/{}_{}'.format(name, wavelet_type))
        plt.show()
        plt.close()
        
    return coefs


def main_QRST(filtered_Data, idx, scored_code, postfix, names, vis=False):
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2','V3','V4','V5','V6','frontal PCA-1', 'I post']
    qrs_chn = 12
    Q_loc, R_loc, S_loc, T_start_loc, T_peak_loc, T_end_loc, filtered_ecg, filtered_ecg2 =extract_QRST(filtered_Data, vis_res=False, vis=False, verbose=False, qrs_chn=qrs_chn,
                   code=str(scored_code) + postfix, 
                   title=str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)

#     fig, axes = plt.subplots(7, 2, figsize=(20,10), sharey=False, sharex=True)
#     lead = 0
#     for j in range(2):
#         for i in range(6):
#             ax = axes[i,j]
#             ax.plot(filtered_Data[lead,:], c='black')
#             ax.set_ylabel(leads[lead])

#             ax.scatter(Q_loc, filtered_Data[lead,:][Q_loc], label='Q', marker='^', s=50)
#             ax.scatter(R_loc, filtered_Data[lead,:][R_loc], label='R', marker='^', s=50)
#             ax.scatter(S_loc, filtered_Data[lead,:][S_loc], label='S', marker='^', s=50)

#             ax.scatter(T_start_loc, filtered_Data[lead,:][T_start_loc], label='T_start', marker='o', s=50)
#             ax.scatter(T_end_loc, filtered_Data[lead,:][T_end_loc], label='T_end', marker='o', s=50)
#             lead += 1

#     for i, ecg in enumerate([filtered_ecg, filtered_ecg2]):
#         axes[6,i].plot(ecg, c='black')
#         sc1 = axes[6,i].scatter(Q_loc, ecg[Q_loc], label='Q', marker='^', s=50)
#         sc2 = axes[6,i].scatter(R_loc, ecg[R_loc], label='R', marker='^', s=50)
#         sc3 = axes[6,i].scatter(S_loc, ecg[S_loc], label='S', marker='^', s=50)

#         sc4 = axes[6,i].scatter(T_start_loc, ecg[T_start_loc], label='T_start', marker='o', s=50)
#         sc5 = axes[6,i].scatter(T_end_loc, ecg[T_end_loc], label='T_end', marker='o', s=50)
#         axes[6,i].set_ylabel(leads[12+i])
        
#         if i == 0:
#             fig.legend([sc1, sc2, sc3, sc4, sc5],     # The line objects
#                ['Q', 'R', 'S', 'T_start', 'T_end'],   # The labels for each line
#                loc="lower center",   # Position of legend
#                borderaxespad=0.1,    # Small spacing around legend box
#                ncol = 5
#                )

#     plt.suptitle(str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)
    
#     if vis:
#         plt.show()
#     else:
#         plt.savefig('../explore/conditions/'+str(scored_code) + postfix)
#     plt.close()
    
    # segment the Q-S segment
    
    fig, axes = plt.subplots(6, 4, figsize=(20,10), sharey=False, sharex=False)
    lead = 0
    for j in range(2):
        for i in range(6):
            ax = axes[i,j]
            ax.set_ylabel(leads[lead])
            for k in range(len(Q_loc)-1):
                ax.plot(filtered_Data[lead, Q_loc[k]:S_loc[k]])
            lead += 1
    lead = 0
    for j in [2,3]:
        for i in range(6):
            ax = axes[i,j]
            ax.set_ylabel(leads[lead])
            for k in range(len(Q_loc)-1):
                ax.plot(filtered_Data[lead, S_loc[k]:Q_loc[k+1]])
            lead += 1  
            
            if lead == 1:
                fig.legend([str(Q_loc[k]) for k in range(len(Q_loc)-1)],   # The labels for each line
                   loc="right",   # Position of legend
                   borderaxespad=0.1,    # Small spacing around legend box
                   ncol = 1
                   )

    plt.suptitle(str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)
    if vis:
        plt.show()
    else:
        plt.savefig('../explore/QRST/'+str(scored_code) + postfix)
    plt.close()    
    