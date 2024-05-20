import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from impl.bin.FBCSP import FBCSP
from impl.bin.Classifier import Classifier
import impl.bin.LoadData as LoadData
from sklearn.svm import SVR
import impl.bin.Preprocess as Preprocess
'''
1. 先init，输入信号频率
2. get_filter_coeff()，根据init里的参数生成滤波参数，不需要输入
3. filter_data(data)，返回滤波后的数据
'''
class FilterBank:
    def __init__(self,fs):
        self.fs = fs # 信号频率
        self.f_trans = 2 
        self.f_width = 4 # 频带宽度
        self.f_pass = np.arange(4,40,self.f_width) #频带 4 8 12 16 ... 36
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    '''
        eeg_data : 3D-ndarray(trials * channels * time)
        window_details : 窗口大小，默认位输入数据的窗口大小
        return : 4D-ndarray(frequency band * trials * channels * time)
    '''
    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((4.5+window_details.get('tmin'))*self.fs):int((4.5+window_details.get('tmax'))*self.fs)+1]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data
