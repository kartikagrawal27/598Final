# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:12:05 2017

@author: Chen-Zhang Xiao
"""
from scipy import stats

activities = ['BenchPress', 'BicepCurl', 'ShoulderFly', 'SeatedPull']
chara_sig=['AX','AYZPC1','GPC1']
Auto_CR_bin=np.zeros(5)
Power_bands=np.zeros(10)

for act in activities:
    for ch in chara_sig:
        sig = Signals[activities[act]][chara_sig[ch]]
        # 5 bins acutocorrelation
        freqs = np.zeros(sig.size)
        freqs = np.fft.rfft(sig)
        R = np.zeros(sig.size)
        R = np.fft.irfft(freqs * np.conj(freqs))
        R_bin_idx = signal.find_peaks_cwt(R,np.arange(1,10))
        if len(R_bin_idx)>=5:
            for i in 5:
                Auto_CR[i] = R[R_bin_idx[i]]
        else:
            print("Autocorrelation bins < 5")
        # RMS
        RMS = np.sqrt(np.mean(sig**2))
        # Power bands * 10
        FFT=np.fft.fft(sig)
        for i in 10:
            Power_bands[i]=FFT[int(sig.size/10)*i]
        # Mean, std, kurtosis, interquartile range
        Mean = np.mean(sig)
        Std = np.std(sig)
        Kurt = stats.kurtosis(sig)
        Iqr = stats.iqr(sig)
        
        
        
        