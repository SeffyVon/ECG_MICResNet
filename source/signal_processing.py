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

def extract_QRST(ecg_12leads, vis=False, vis_res=False, only_qrs=False, verbose=False,
                 filter_twice=False, qrs_chn=10, code=None, title=None):
    """
    To extract the QRST.
    Cannot cope with the negative R yet!
    """
    # transverse plane
    #transverse_leads = PCA(2).fit_transform(ecg_12leads[6:,:].transpose()).transpose() # ecg_12leads[[7, 12],:]
    # frontal plane
    frontal_leads = PCA(2).fit_transform(ecg_12leads[:6,:].transpose()).transpose() # ecg_12leads[[0, 5],:]
    # 3D space
    #three_leads =  ecg_12leads[[0, 5, 7],:] #PCA(3).fit_transform(ecg_12leads.transpose()).transpose() #
    
    filtered_ecg = ecg_12leads[qrs_chn,:] if qrs_chn != 12 else \
        np.sqrt(frontal_leads[0,:]**2 + frontal_leads[1,:]**2) * np.sign(frontal_leads[0,:]) * np.sign(frontal_leads[1,:])

    if vis:    
        plt.figure(figsize=(15,3))
        plt.plot(filtered_ecg)
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
    
    GAP_SINCE_S = 10 # 500 Hz, change 100 to 50
    #T_start_loc = S_loc + GAP_SINCE_S 
    T_peak_loc = np.zeros(len(T_start_loc)-1, dtype=int)
    T_end_loc = np.zeros(len(T_start_loc)-1, dtype=int)
    time_vars = np.zeros(ecg_12leads.shape[1])
    
    
    for i in range(len(T_start_loc)-1):

        # from this T_start_loc to next Q
        time_vars[T_start_loc[i]+GAP_SINCE_S:Q_loc[i+1]] = filtered_ecg[T_start_loc[i]+GAP_SINCE_S:Q_loc[i+1]]
         
        # T peak is defined within the start to the 2/3 of the S-Q segment
        T_peak_bound = int(1/3 * T_start_loc[i] + 2/3 * Q_loc[i+1]) 
        if T_peak_bound <= T_start_loc[i]:
            T_end_loc[i] = T_start_loc[i]
            continue
        if verbose:
            print("T_start_loc[i]", T_start_loc[i], "T_peak_bound", T_peak_bound)
            
        T_peak_loc[i] = np.argmax(time_vars[T_start_loc[i]:T_peak_bound]) + T_start_loc[i]
        if verbose:
            print("T_peak_loc", T_peak_loc[i])

        # backward search for the end of the T end
        T_end_loc[i] = T_peak_loc[i]
        while T_end_loc[i]<Q_loc[i+1] and (time_vars[T_end_loc[i]+1]<time_vars[T_end_loc[i]]):
            T_end_loc[i] = T_end_loc[i] + 1
        if verbose:
            print("T_end_loc", T_end_loc[i])
            
    if vis or vis_res:    
        plt.figure(figsize=(20,10))
        plt.plot(filtered_ecg, label="filtered ecg")
        plt.plot(time_vars,  '--', label="time variance")
        plt.scatter(Q_loc, filtered_ecg[Q_loc], label='Q', s=100)
        plt.scatter(R_loc, filtered_ecg[R_loc], label='R', s=100)
        plt.scatter(S_loc, filtered_ecg[S_loc], label='S', s=100)
        plt.scatter(T_start_loc, filtered_ecg[T_start_loc], label='T start', marker='^', s=100)
        plt.scatter(T_peak_loc, filtered_ecg[T_peak_loc], label='T peak', marker='^', s=100)
        plt.scatter(T_end_loc, filtered_ecg[T_end_loc], label='T end', marker='^', s=100)
        plt.legend(loc='best')
        if title is not None:
            plt.title(title) 
        if code is not None:
            plt.savefig('../explore/QRST/'+str(code))
        plt.show()
        plt.close()
  
    return Q_loc, R_loc, S_loc, T_start_loc, T_peak_loc, T_end_loc


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