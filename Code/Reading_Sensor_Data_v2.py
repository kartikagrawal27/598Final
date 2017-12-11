# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:40:25 2017

@author: Chen-Zhang Xiao
"""
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict
from sklearn.decomposition import FastICA, PCA


def infinidict():
    return defaultdict(infinidict)


def VerySimplePeakdetection(t, signal, case):
    if (case == 1):
        max_peak = []
        max_val = -np.inf
        max_p = 0
        N = len(signal)
        for i in range(1, N - 1):
            if signal[i] > max_val:
                max_val = signal[i]
                max_p = t[i]
        max_peak = [max_p, max_val]
        return np.array(max_peak)
    else:
        min_peak = []
        min_val = np.inf
        min_p = 0
        N = len(signal)
        for i in range(1, N - 1):
            if signal[i] < min_val:
                min_val = signal[i]
                min_p = t[i]
        min_peak = [min_p, min_val]
        return np.array(min_peak)


def SimplePeakdetection(t, signal):
    max_peak = [];
    min_peak = [];
    max_p = 0;
    min_p = 0;
    N = len(signal)
    for i in range(10, N - 10):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            max_val = signal[i]
            max_p = t[i]
            max_peak.append([max_p, max_val])
    return np.array(max_peak)


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
truncations['BicepCurl']['t1'] = (0, -1)
truncations['BicepCurl']['t2'] = (134, -1)
truncations['BicepCurl']['t3'] = (0, -1)
truncations['BicepCurl']['t4'] = (0, -1)
truncations['ShoulderFly']['t1'] = (145, -1)
truncations['ShoulderFly']['t2'] = (0, -1)
truncations['ShoulderFly']['t3'] = (0, -1)
truncations['ShoulderFly']['t4'] = (0, -1)
truncations['SeatedPull']['t1'] = (90, 1290)
truncations['SeatedPull']['t2'] = (77, 1455)
truncations['SeatedPull']['t3'] = (87, 1302)
truncations['SeatedPull']['t4'] = (96, 1361)

truncations

for activity in activities:
    ax = []
    ay = []
    az = []
    gx = []
    gy = []
    gz = []
    for trial in trials[0:4]:
        temp_ax = [i for i in (data[activity][trial]['accelerometer']['x'])]
        temp_ay = [i for i in (data[activity][trial]['accelerometer']['y'])]
        temp_az = [i for i in (data[activity][trial]['accelerometer']['z'])]
        temp_gx = [i for i in (data[activity][trial]['gyroscope']['x'])]
        temp_gy = [i for i in (data[activity][trial]['gyroscope']['y'])]
        temp_gz = [i for i in (data[activity][trial]['gyroscope']['z'])]

        ax = ax + temp_ax[truncations[activity][trial][0]:truncations[activity][trial][1]]
        ay = ay + temp_ay[truncations[activity][trial][0]:truncations[activity][trial][1]]
        az = az + temp_az[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gx = gx + temp_gx[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gy = gy + temp_gy[truncations[activity][trial][0]:truncations[activity][trial][1]]
        gz = gz + temp_gz[truncations[activity][trial][0]:truncations[activity][trial][1]]
        
    signals[activity]['AX'] = ax
    signals[activity]['AY'] = ay
    signals[activity]['AZ'] = az
    signals[activity]['GX'] = gx
    signals[activity]['GY'] = gy
    signals[activity]['GZ'] = gz
        
for activity in activities:
    signal_accel = np.array([signals[activity]['AX'], signals[activity]['AY'], signals[activity]['AZ']])
    signal_gyroscope = np.array([signals[activity]['GX'], signals[activity]['GY'], signals[activity]['GZ']])

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

    signals[activity]['AYZPC1'] = PC_YZacc
    signals[activity]['GPC1'] = PC_gyroscope

print(signals)


