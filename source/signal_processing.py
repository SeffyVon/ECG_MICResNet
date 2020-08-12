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


def MA(x, n):
    # Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    h = np.ones(n)/n
    y = signal.convolve(x, h)
    y = y[(n//2):]
    y = y/max(abs(y))
    return y

from sklearn.decomposition import PCA

def adjust_Q_and_S(Q_p, S_p, ecgs, R_p): # 100mV 0.002 ms
    
    Q_p_res = []
    S_p_res = []
    for ecg in ecgs:
        diff = np.diff(ecg)
        # Adjusting Q_loc
        thresh = np.abs(ecg[R_p]) * 0.1
        while Q_p-1 > 0 and ecg[Q_p-1] < ecg[Q_p] and diff[Q_p] > thresh:
            Q_p = Q_p - 1
            
        Q_p_res.append(Q_p)

        # Adjusting S_loc
        while S_p+1 < len(ecg) and ecg[S_p+1] < ecg[S_p] and diff[S_p] < -thresh:
            S_p = S_p+1
            
        S_p_res.append(S_p)
            
    #print("Q_p, S_p", Q_p_res, S_p_res)
   
    return np.max(Q_p_res), np.min(S_p_res)

from sklearn import mixture 
def Pan_Tompkins_QRS(ecg, ecg2, ecg_12leads, verbose=False):
    """
    Use Pan Tompkins algorithm to extract QRS
    cannot cope with negative R yet!
    :param ecg:
    :param verbose:
    :return:
    """
    
    def PT_preprocessing(ecg):
        ecg = ecg/max(abs(ecg))
        differentiated_ecg = derivative(ecg)
        squaring_ecg = squaring(differentiated_ecg)
        return squaring_ecg
    
    squaring_ecg = PT_preprocessing(ecg)
    integrated_ecg = MA(squaring_ecg, n=51) #30 for 200Hz, we have 500Hz so we choose 30*2.5 = 75

    peak_indices,_ = find_peaks(integrated_ecg, distance=100)
    if len(peak_indices) < 3:
        print("!!!!!!!")
        peak_indices,_ = find_peaks(integrated_ecg)
    
    # use 2 Gaussian Mixture Regression to fit
    gmm = mixture.GaussianMixture(random_state=0,
                              n_components=2,
                                 n_init=10)
    
    membership = gmm.fit_predict(integrated_ecg[peak_indices].reshape((-1,1))).flatten()
    big_cluster = np.argmax(gmm.means_)
    big_members = peak_indices[np.argwhere(membership==big_cluster)].flatten()
    if verbose:
        print("membership", membership)
        print("big_members", big_members)
    
    time = 0
    RR_TH = 2000
    while len(big_members) < 3 or np.max(np.diff(big_members))>RR_TH or big_members[0] > RR_TH or len(integrated_ecg) - big_members[-1] > RR_TH or len(big_members)/len(peak_indices)<0.05 or big_members[-1] - big_members[0] < 300: 
        time += 1
        if time > 3:
            break
        peak_indices = list(peak_indices)
        [peak_indices.remove(big_member) for big_member in big_members]
        gmm = mixture.GaussianMixture(random_state=0,
                              n_components=2,
                              n_init=10)
        membership = gmm.fit_predict(integrated_ecg[peak_indices].reshape((-1,1))).flatten()
        big_cluster = np.argmax(gmm.means_)
        big_members_idx = np.argwhere(membership==big_cluster).flatten()
        big_members = np.array(peak_indices)[big_members_idx] if len(big_members_idx)>0 else [np.array(peak_indices)[np.argmax(integrated_ecg[peak_indices])]]
        if verbose:
            print("removed!!!!!!")
            print("membership", membership)
            print("big_members", big_members)
        

    peak_indices = np.array(peak_indices)
    big_cluster = np.argmax(gmm.means_)
    
    gmm = mixture.GaussianMixture(random_state=0,
                              n_components=2,
                                 n_init=10)
    membership = gmm.fit_predict(integrated_ecg[peak_indices].reshape((-1,1))).flatten()
    big_members_idx = np.argwhere(membership==big_cluster).flatten()
    big_members = np.array(peak_indices)[big_members_idx] if len(big_members_idx)>0 else np.array(peak_indices)[np.argmax(integrated_ecg[peak_indices])]
    thresh = np.min(integrated_ecg[big_members]*0.99)

    if verbose:
        
        plt.hist(integrated_ecg[peak_indices])
        plt.show()
        plt.close()
        
        print("membership", membership)
        print("gmm.means_", gmm.means_)
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

    if left[0] < 100 and len(left)>1:
        left = left[1:]
        right = right[1:]

    if left[-1] > len(ecg) - 100:
        left = left[:-1]
        right = right[:-1]

    # filter too near left and right
#     left_filtered = []
#     right_filtered = []
#     RR_TH = 50
#     i = 0
#     while i < len(left)-1:
#         if right[i+1] - left[i] <= RR_TH:
#             left_filtered.append(left[i])
#             right_filtered.append(right[i+1])
#             i+=1
#         else:
#             left_filtered.append(left[i])
#             right_filtered.append(right[i])            
#         i+=1
#     left = left_filtered
#     right = right_filtered

    # scan from left to right
    R_loc=np.zeros(len(left), dtype=int)
    Q_loc=np.zeros(len(left), dtype=int)
    S_loc=np.zeros(len(left), dtype=int)

    # use a copy of filtered_ecg to record "un-inverted" filtered_ecg
    segment = ecg.copy()
    segment2 = ecg2.copy()
    
    for lead in range(12):
        ecg_12leads[lead,:] = correct_axis_ecg(ecg_12leads[lead,:])
        
    for i in range(len(left)):
        
        segment_ecg = correct_axis_ecg(ecg[left[i]:right[i]+1])

        segment[left[i]:right[i]+1] = segment_ecg

        R_loc[i] = np.argmax(segment[left[i]:right[i]+1]) + left[i]
    
        # R-left and R_right to search for the max segment as R
        # R_left
        R_left = R_loc[i]
        while R_left-1 > left[i] and segment[R_left-1] > segment[R_left]:
            R_left = R_left-1
        if segment[R_left] > segment[R_loc[i]]:
            R_loc[i] = R_left

        # R_right
        R_right = R_loc[i]            
        while R_right+1 < right[i] and segment[R_right+1] > segment[R_right]:
            R_right = R_right+1
        if segment[R_right] > segment[R_loc[i]]:
            R_loc[i] = R_right

        left[i] = min(left[i], R_loc[i])
        right[i] = max(right[i], R_loc[i])

        Q_loc[i] = np.argmin(ecg[left[i]:R_loc[i]+1]) + left[i]# left - R, R_loc[i]-10 #
        S_loc[i] = np.argmin(ecg[R_loc[i]:right[i]+1]) + R_loc[i] # R-right R_loc[i]+10 #

        Q_loc[i], S_loc[i] = adjust_Q_and_S(Q_loc[i], S_loc[i], ecg_12leads, R_loc[i])

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

def correct_axis_ecg(ecg, vis=False):
    peak_indices,_ = find_peaks(np.abs(ecg), height=np.percentile(np.abs(ecg), 98))
    if vis:
        plt.plot(ecg)
        plt.scatter(peak_indices, ecg[peak_indices])
        plt.show()
        plt.close()
    sign = 1 if np.sum(ecg[peak_indices] > 0) > np.sum(ecg[peak_indices] < 0) else -1
    ecg *= sign
    return ecg
        
def extract_QRST(ecg_12leads, vis=False, vis_res=False, only_qrs=True, verbose=False,
                 filter_twice=False, qrs_chn=12, code=None, title=None
                ):
    """
    To extract the QRST.
    Cannot cope with the negative R yet!
    
    """

    filtered_ecg = None
    filtered_ecg2 = None
    if qrs_chn == 12:
        # frontal leads
        pca_res = PCA(2).fit_transform(ecg_12leads[[0,5,7],:].transpose())
        filtered_ecg = pca_res[:,0].flatten()
        filtered_ecg2 = pca_res[:,1].flatten()
        
        filtered_ecg = correct_axis_ecg(filtered_ecg)
        filtered_ecg2 = correct_axis_ecg(filtered_ecg2)
    else:
        filtered_ecg = ecg_12leads[qrs_chn,:]

    
    if vis:    
        plt.figure(figsize=(15,3))
        #plt.plot(frontal_leads.transpose())
        plt.plot(filtered_ecg, c='black')
        plt.title('signal for QRST segmentation')
        plt.show()
        plt.close()
    
#     thresh = 0.15
    Q_loc, R_loc, S_loc, integrated_ecg, left, right, thresh = Pan_Tompkins_QRS(filtered_ecg, filtered_ecg2, ecg_12leads, verbose=verbose)
#     while thresh >= 0.02 and len(R_loc) < 3 or np.std([R_loc[i+1]-R_loc[i] for i in range(len(R_loc)-1)]) > 300 or R_loc[-1]<1400 or R_loc[0] > 1600:
#         thresh  -= 0.01
#         Q_loc, R_loc, S_loc, integrated_ecg, left, right = Pan_Tompkins_QRS(filtered_ecg, filtered_ecg2, ecg_12leads, verbose=verbose, thresh=thresh)
       
    if verbose:
        print("------------------")
        print("------report------")
        print("RR std", np.std([R_loc[i+1]-R_loc[i] for i in range(len(R_loc)-1)]))
        print("R_loc", R_loc)
        print("------------------")
        print("------------------")
        
    if vis:    
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        
        ax.plot(filtered_ecg, '--', label='filtered_ecg', c='grey')
        ax2 = ax.twinx()
        ax2.hlines(thresh, 0, len(filtered_ecg))
        ax2.plot(integrated_ecg, label='integrated signal')
        ax.scatter(left, integrated_ecg[left], label='left')
        ax.scatter(right, integrated_ecg[right], label='right')
        ax.scatter(Q_loc, filtered_ecg[Q_loc], label='Q', marker='^', s=100)
        ax.scatter(R_loc, filtered_ecg[R_loc], label='R', marker='^', s=100)
        ax.scatter(S_loc, filtered_ecg[S_loc], label='S', marker='^', s=100)
        plt.legend(loc='best')
        #plt.title("Detect QRS of patient {} phase {}".format(patient_num, phase_num))
        #plt.savefig('../result/pAF/seg/' + str(patient_num) + '_' + str(phase_num) + '_seg.png')
        plt.show()
        plt.close()
        
    if only_qrs:
        return Q_loc, R_loc, S_loc, None, None, None, filtered_ecg, filtered_ecg2

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
    
    RR = np.median([R_loc[i+1]-R_loc[i] for i in range(len(R_loc)-1)])
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
                    T_peak_loc[i] = np.argmax(time_vars[T_start_loc[i]: T_end_loc[i]]) + T_start_loc[i]
                    if vis:
                        plt.axvspan(grouping[0], grouping[-1], alpha=0.4, color='C3')
#                 elif grouping[-1]- R_loc[i] > RTmax:
#                     if vis:
#                         plt.axvspan(grouping[0], grouping[-1], alpha=0.4, color='C4')                

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

import seaborn as sns
sns.set_style("whitegrid")

def main_QRST(filtered_Data, idx, scored_code, postfix, names, vis=False, verbose=False, fig1=False, fig2=False):
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2','V3','V4','V5','V6','PCA-1','', 'PCA-2']
    qrs_chn = 12
    Q_loc, R_loc, S_loc, T_start_loc, T_peak_loc, T_end_loc, filtered_ecg, filtered_ecg2 =extract_QRST(filtered_Data, vis_res=False, vis=vis, verbose=verbose, qrs_chn=qrs_chn,
                   code=str(scored_code) + postfix, 
                   title=str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)

    if fig1:
        fig, axes = plt.subplots(7, 2, figsize=(20,10), sharey=False, sharex=True)
        lead = 0
        for j in range(2):
            for i in range(6):
                ax = axes[i,j]
                ax.plot(filtered_Data[lead,:], c='black')
                ax.set_ylabel(leads[lead])

                ax.scatter(Q_loc, filtered_Data[lead,:][Q_loc], label='Q', marker='^', s=50)
                ax.scatter(R_loc, filtered_Data[lead,:][R_loc], label='R', marker='^', s=50)
                ax.scatter(S_loc, filtered_Data[lead,:][S_loc], label='S', marker='^', s=50)

               # ax.scatter(T_start_loc, filtered_Data[lead,:][T_start_loc], label='T_start', marker='o', s=50)
               # ax.scatter(T_end_loc, filtered_Data[lead,:][T_end_loc], label='T_end', marker='o', s=50)
                lead += 1

        for i, ecg in enumerate([filtered_ecg, filtered_ecg2]):
            axes[6,i].plot(ecg, c='black')
            sc1 = axes[6,i].scatter(Q_loc, ecg[Q_loc], label='Q', marker='^', s=50)
            sc2 = axes[6,i].scatter(R_loc, ecg[R_loc], label='R', marker='^', s=50)
            sc3 = axes[6,i].scatter(S_loc, ecg[S_loc], label='S', marker='^', s=50)

          #  sc4 = axes[6,i].scatter(T_start_loc, ecg[T_start_loc], label='T_start', marker='o', s=50)
          #  sc5 = axes[6,i].scatter(T_end_loc, ecg[T_end_loc], label='T_end', marker='o', s=50)
            axes[6,i].set_ylabel(leads[12+i])

            if i == 0:
                fig.legend([sc1, sc2, sc3],     # The line objects
                   ['Q', 'R', 'S'],   # The labels for each line
                   loc="lower center",   # Position of legend
                   borderaxespad=0.1,    # Small spacing around legend box
                   ncol = 5
                   )

        plt.suptitle(str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)

        if vis:
            plt.show()
        else:
            plt.savefig('../explore/conditions/'+str(idx) + postfix)
        plt.close()
    
    if fig2:
        RR_avg = np.median([R_loc[k+1] - R_loc[k] for k in range(len(Q_loc)-1)])
        RR_th = (0.3 * RR_avg, 3 * RR_avg)
        if verbose:
            print("RR_avg", RR_avg, "RR_th", RR_th)

        # segment the Q-S segment
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(7, 4, figsize=(20,18), sharey=False, sharex=False)
        lead = 0


        ks = [k for k in range(len(Q_loc)-1) if R_loc[k+1] - R_loc[k] > RR_th[0] 
              and R_loc[k+1] - R_loc[k] < RR_th[1]
              and S_loc[k] - Q_loc[k] > 10 and R_loc[k] != Q_loc[k] and R_loc[k] != S_loc[k]] # 10 / 500 = 0.02s
        #print("ks", ks)
        n = len(ks)
        colors = plt.cm.jet(np.linspace(0,1,n))
        for j in [0,2]:
            for i in range(6):
                ax = axes[i,j]
                ax.set_ylabel(leads[lead])
                times = [range(Q_loc[k]-R_loc[k],S_loc[k]-R_loc[k]) for k in ks]
                lines = [filtered_Data[lead, Q_loc[k]:S_loc[k]] for k in ks]
                for k, line in enumerate(lines):
                    ax.plot(times[k], line, c=colors[k])

                ax = axes[i,j+1]
                ax.set_ylabel(leads[lead])
                lines = [filtered_Data[lead, S_loc[k]:Q_loc[k+1]] for k in ks]
                for k, line in enumerate(lines):
                    ax.plot( line, c=colors[k])

                lead += 1  

                if lead == 1:
                    fig.legend([str(Q_loc[k]) for k in range(len(Q_loc)-1)],   # The labels for each line
                       loc="right",   # Position of legend
                       borderaxespad=0.1,    # Small spacing around legend box
                       ncol = 1,
                       title='Q time'
                       )

        j = 0
        for ecg in [filtered_ecg, filtered_ecg2]: # 0, 1 => col: 0,1; 2,3. label: 12, 13

            times = [range(Q_loc[k]-R_loc[k],S_loc[k]-R_loc[k]) for k in ks]
            lines = [ecg[Q_loc[k]:S_loc[k]] for k in ks]   
            for k, line in enumerate(lines):
                axes[6,j].plot(times[k], line, c=colors[k])
            axes[6,j].set_ylabel(leads[12+j])

            lines = [ecg[S_loc[k]:Q_loc[k+1]] for k in ks]    
            for k, line in enumerate(lines):
                axes[6,j+1].plot( line, c=colors[k])
            axes[6,j+1].set_ylabel(leads[12+j])
            
            j+=2

        plt.suptitle(str(idx) + ' ' + str(scored_code) + ' ' + str(leads[qrs_chn]) + ' ' + names)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
        if vis:
            plt.show()
        else:
            plt.savefig('../explore/QRST/'+str(idx) + postfix)
        plt.close()    
        
    return Q_loc
    