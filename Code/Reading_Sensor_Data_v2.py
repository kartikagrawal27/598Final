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

def VerySimplePeakdetection(t,signal,case):
    if (case ==1):
        max_peak=[]
        max_val=-np.inf
        max_p=0
        N=len(signal)
        for i in range (1,N-1):
            if signal[i]>max_val:
                max_val=signal[i]
                max_p=t[i]
        max_peak=[max_p,max_val]
        return np.array(max_peak)
    else:
        min_peak=[]
        min_val=np.inf
        min_p=0
        N=len(signal)
        for i in range (1,N-1):
            if signal[i]<min_val:
                min_val=signal[i]
                min_p=t[i]
        min_peak=[min_p,min_val]
        return np.array(min_peak)
    
def SimplePeakdetection(t,signal):
    max_peak=[];
    min_peak=[];
    max_p=0;
    min_p=0;
    N=len(signal)
    for i in range (10,N-10):
        if signal[i]<signal[i-1] and signal[i]<signal[i+1]:
            max_val=signal[i]
            max_p=t[i]
            max_peak.append([max_p,max_val])
    return np.array(max_peak)

    
activities = ['BenchPress', 'BicepCurl', 'ShoulderFly', 'SeatedPull']
trials = ['t1', 't2', 't3', 't4', 't5']
sensors = ['accelerometer', 'gyroscope','gravity', 'linear_acceleration']
axis = ['x', 'y', 'z']

data = infinidict()

for activity in activities:
	for trial in trials:
		for sensor in sensors:
			my_path = os.path.abspath(os.path.dirname(__file__))
			path = os.path.join(my_path, "SensorData",str(activity),"Sub01" + str(activity) + "_" + str(sensor) + "_" + str(trial) +".txt")
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
				xs.append(x)
				ys.append(y)
				zs.append(z) 
			data[str(activity)][str(trial)][str(sensor)]['x'] = xs
			data[str(activity)][str(trial)][str(sensor)]['y'] = ys
			data[str(activity)][str(trial)][str(sensor)]['z'] = zs
#generating subplot for an example data, comparing acc & gyro for four activities
#using hamming window of 64 
#plt.figure(1)
#for i in range(1,4):
#    ax = plt.subplot(3,1,i)
#    ax.plot(S[:,i])
#plt.show
#

#
#plt.figure(3)
#plt.plot(Sxx)
act=3
tr=1

ax=[float(i) for i in (data[activities[act]][trials[tr]][sensors[0]]['x'])]
ay=[float(i) for i in (data[activities[act]][trials[tr]][sensors[0]]['y'])]
az=[float(i) for i in (data[activities[act]][trials[tr]][sensors[0]]['z'])]
gx=[float(i) for i in (data[activities[act]][trials[tr]][sensors[1]]['x'])]
gy=[float(i) for i in (data[activities[act]][trials[tr]][sensors[1]]['y'])]
gz=[float(i) for i in (data[activities[act]][trials[tr]][sensors[1]]['z'])]
#convert data from string to array
siga=np.array([ax,ay,az])
sigg=np.array([gx,gy,gz])
N=np.minimum(sigg.shape[1],siga.shape[1])
if sigg.shape[1]> siga.shape[1]:
    sigg=np.delete(sigg,-1,1)
if siga.shape[1]> sigg.shape[1]:
    siga=np.delete(siga,-1,1)
    
#low pass filter
b, a = signal.butter(5,1/25)
fa = signal.filtfilt(b, a, siga)
fg = signal.filtfilt(b, a, sigg)
ft=np.concatenate((fa,fg))

# filtered acc data
plt.figure(1)
plt.plot(fa.T)
plt.show
# filtered gyro data
plt.figure(2)
plt.plot(fg.T)
plt.show

#dimension reduciton with PCA
pca = PCA(n_components=1)
PC = pca.fit_transform(fg.T)  # Reconstruct signals based on orthogonal components
plt.figure(3)
plt.plot(PC)
plt.show

#dimension reduciton with ICA
ica = FastICA(n_components=1)
IC = ica.fit_transform(fg.T)  # Reconstruct signals based on orthogonal components
plt.figure(4)
plt.plot(IC)
plt.show

#plt.figure(5)
#fs=50
#window=signal.get_window('hamming', 64)
#F, t, Sxx = signal.spectrogram(IC[:,0],fs,window,64)
#plt.pcolormesh(t, F, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

# perform autocorrelation
f=PC[:,0]

subframe_size=int(8)
numFrames = int(siga.shape[1] / subframe_size)
FRAME_SIZE = int(64)
numsubframe = int(FRAME_SIZE/subframe_size)

fs=50
N=siga.shape[1]
t = np.linspace(0,N-1,N)#np.linspace(0,N/fs,N)
t_c = np.linspace(0,999,1000)

# Building filter for human motion data
nyq= fs/2 # nyquist frequency
desired=(0,0,1,1,0,0)
bands=(0,0.1,0.2,2,2.1,25)
b = signal.firls(61,bands,desired,nyq=nyq)

# calculating cross-correlation

nump_peaks = []
max_r=[]
max_R=[]
std_R=[]
max_peaks=[]
std_peaks=[]
is_working_flag=0
working_peak=[]
peak_sig=np.zeros(FRAME_SIZE)
peak_frame=[]
segment=[]
R_mean=[]
R_diff=[]

# Looking into each frame
for i in range (0, numFrames):
    
    # build frame based on subframes
    sig = f[(i)*subframe_size:(i+1)*subframe_size]
    ts = t[(i)*subframe_size:(i+1)*subframe_size]
    
    # cross-correlation
    freqs = np.zeros(sig.size)
    freqs = np.fft.rfft(sig)
    R=np.zeros(sig.size)
    R=np.fft.irfft(freqs * np.conj(freqs))
    if i==0:
        R_mean.append(np.mean(R))
    else:
        R_mean.append(0.2*np.mean(R)+0.8*R_mean[-1])
        R_diff.append(np.mean(R)-R_mean[-1])
        
    
#    plt.figure(i+10)
#    plt.plot(sig)
#    plt.plot(R)
#    plt.show()
    indexs = signal.find_peaks_cwt(R,np.arange(1,10))
    nump_peaks.append(len(indexs))
    if len(indexs)>0:
        max_R.append(np.amax(R[indexs]))
        std_R.append(np.std(R[indexs]))
    else:
        max_R.append(0)
        std_R.append(0)

    # find characteristics of cc R 
#    max_peaks = np.max
#    
#    


plt.figure(80)
plt.plot(R_diff) # use R_diff to segment the actual signal
plt.show()
