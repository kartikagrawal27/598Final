# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:36:08 2017

@author: Chen-Zhang Xiao
"""

# main file
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy import stats

def infinidict():
    return defaultdict(infinidict)

def Local_peakdetection(signal,ahead,case):
    if (case ==1):
        peak=[]
        peak_val=0
        peak_loc=0
        N=len(signal)
        for i in range (ahead,N-ahead):
            if signal[i]>signal[i-1] and signal[i]>signal[i+1] and signal[i]>signal[i-ahead] and signal[i]>signal[i+ahead]:
                peak_val=signal[i]
                peak_loc=i
                peak.append(peak_val)
        return peak
    else:
        peak=[]
        peak_val=0
        peak_loc=0
        N=len(signal)
        for i in range (ahead,N-ahead):
            if signal[i]<signal[i-1] and signal[i]<signal[i+1] and signal[i]<signal[i-ahead] and signal[i]<signal[i+ahead]:
                peak_val=signal[i]
                peak_loc=i
                peak.append(peak_val)
        return peak

activities = ['BenchPress', 'BicepCurl', 'ShoulderFly', 'SeatedPull']
trials = ['t1', 't2', 't3', 't4', 't5']
sensors = ['accelerometer', 'gyroscope', 'gravity', 'linear_acceleration']
axis = ['x', 'y', 'z']

data = infinidict()

for activity in activities:
    for trial in trials:
        for sensor in sensors:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../SensorData", str(activity),
                                "Sub01" + str(activity) + "_" + str(sensor) + "_" + str(trial) + ".txt")
            file = open(path, 'r')

            lines = []
            xs = []
            ys = []
            zs = []
            for line in file:
                lines.append(line)

            for i in range(13, len(lines)):
                myLine = lines[i]
                _, x, y, z = myLine.split()
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            data[str(activity)][str(trial)][str(sensor)]['x'] = xs
            data[str(activity)][str(trial)][str(sensor)]['y'] = ys
            data[str(activity)][str(trial)][str(sensor)]['z'] = zs

signals = infinidict()

properties = ['AX', 'AYZPC1', 'GPC1']
truncations = infinidict()

truncations['BenchPress']['t1'] = (131, 1374)
truncations['BenchPress']['t2'] = (94, 1370)
truncations['BenchPress']['t3'] = (110, 1415)
truncations['BenchPress']['t4'] = (144, 1571)
truncations['BenchPress']['t5'] = (114, 1530)
truncations['BicepCurl']['t1'] = (0, -1)
truncations['BicepCurl']['t2'] = (134, -1)
truncations['BicepCurl']['t3'] = (0, -1)
truncations['BicepCurl']['t4'] = (0, -1)
truncations['BicepCurl']['t5'] = (0, -1)
truncations['ShoulderFly']['t1'] = (145, -1)
truncations['ShoulderFly']['t2'] = (0, -1)
truncations['ShoulderFly']['t3'] = (0, -1)
truncations['ShoulderFly']['t4'] = (0, -1)
truncations['ShoulderFly']['t5'] = (0, -1)
truncations['SeatedPull']['t1'] = (90, 1290)
truncations['SeatedPull']['t2'] = (77, 1455)
truncations['SeatedPull']['t3'] = (87, 1302)
truncations['SeatedPull']['t4'] = (96, 1361)
truncations['SeatedPull']['t5'] = (72, 1250)
truncations

for activity in activities:
    ax = []
    ay = []
    az = []
    gx = []
    gy = []
    gz = []
    for trial in trials:
        temp_ax = [i for i in (data[activity][trial]['accelerometer']['x'])]
        temp_ay = [i for i in (data[activity][trial]['accelerometer']['y'])]
        temp_az = [i for i in (data[activity][trial]['accelerometer']['z'])]
        temp_gx = [i for i in (data[activity][trial]['gyroscope']['x'])]
        temp_gy = [i for i in (data[activity][trial]['gyroscope']['y'])]
        temp_gz = [i for i in (data[activity][trial]['gyroscope']['z'])]

        ax = temp_ax[truncations[activity][trial][0]:truncations[activity][trial][1]]
        ay = temp_ay[truncations[activity][trial][0]:truncations[activity][trial][1]]
        az = temp_az[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gx = temp_gx[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gy = temp_gy[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gz = temp_gz[truncations[activity][trial][0]:truncations[activity][trial][1]]
        
        signals[activity][trial]['AX'] = ax
        signals[activity][trial]['AY'] = ay
        signals[activity][trial]['AZ'] = az
        signals[activity][trial]['GX'] = gx
        signals[activity][trial]['GY'] = gy
        signals[activity][trial]['GZ'] = gz


        signal_accel = np.array([ax, ay, az])
        signal_gyroscope = np.array([gx, gy, gz])
    
        aYZPC1_temp = signal_accel[1:2, :]
    
        N = np.minimum(signal_gyroscope.shape[1], signal_accel.shape[1])
        if signal_gyroscope.shape[1] > signal_accel.shape[1]:
            signal_gyroscope = np.delete(signal_gyroscope, -1, 1)
        if signal_accel.shape[1] > signal_gyroscope.shape[1]:
            signal_accel = np.delete(signal_accel, -1, 1)
    
        b, a = signal.butter(5, 1.5 / 25)
        filtered_accel_signal = signal.filtfilt(b, a, signal_accel)
        filtered_gyro_signal = signal.filtfilt(b, a, signal_gyroscope)
        filtered_accel_YZsignal = signal.filtfilt(b,a , signal_accel[1:2, :])
    
        # dimension reduciton with PCA
        pca = PCA(n_components=1)
        PC_gyroscope = pca.fit_transform(filtered_gyro_signal.T)  # Reconstruct signals based on orthogonal components
    
        pca = PCA(n_components=1)
        PC_YZacc = pca.fit_transform(filtered_accel_YZsignal.T)  # Reconstruct signals based on orthogonal components
    
        signals[activity][trial]['AYZPC1'] = PC_YZacc
        signals[activity][trial]['GPC1'] = PC_gyroscope



main_features=[]
for activity in activities:
    for trial in trials[0:4]:
        trial_features=[]
        for prop in properties:
            R_bin=[]
            Power_bands=[]
            sig=signals[activity][trial][prop]
            sig=np.array(sig)
            # 5 bins acutocorrelation
            R=signal.correlate(sig,sig,"same")
            # need better peak detection method
            R_bin_peak=Local_peakdetection(sig,30,1)  

            if len(R_bin_peak)>=10:
                for i in range (0,5):
                    R_bin.append(R_bin_peak[2*i])
            else:
                print("Autocorrelation bins < 5")
            # RMS
            RMS = np.sqrt(np.mean(sig**2))
            # Power bands * 10
            FFT=np.fft.fft(sig)
            for i in range (0,10):
                Power_bands.append(abs(FFT[int(sig.size/10)*i]))
            # Mean, std, kurtosis, interquartile range
            Mean = np.mean(sig)
            Std = np.std(sig)
            Kurt = stats.kurtosis(sig)
            Iqr = stats.iqr(sig)

            prop_features = []

            for val in R_bin:
                prop_features.append(val)
            prop_features.append(RMS)
            prop_features = prop_features + Power_bands
            prop_features.append(Mean)
            prop_features.append(Std)
            prop_features.append(val)
            prop_features.append(Iqr)
            trial_features = trial_features + prop_features
            print(len(trial_features))
        main_features.append(trial_features)

        
        

