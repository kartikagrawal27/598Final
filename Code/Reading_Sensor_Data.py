# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:40:25 2017

@author: Chen-Zhang Xiao
"""
import numpy as np
from scipy import signal
import os
import os.path
from collections import defaultdict

def infinidict():
    return defaultdict(infinidict)

# my_path = os.path.abspath(os.path.dirname(__file__))
# path = os.path.join(my_path, "../SensorData/Bench Press/Sub01BenchPress_accelerometer_t1.txt")

activities = ['Bench Press', 'Bicep Curl', 'Shoulder Fly', 'Seated Pull']
trials = ['t1', 't2', 't3', 't4', 't5']
sensors = ['accelerometer', 'gravity','gyroscope', 'linear_acceleration']
axis = ['x', 'y', 'z']

data = infinidict()

for activity in activities:
	for trial in trials:
		for sensor in sensors:
			my_path = os.path.abspath(os.path.dirname(__file__))
			path = os.path.join(my_path, "../SensorData/"+ str(activity) +"/Sub01" + str(activity.replace(" ", "")) + "_" + str(sensor) + "_" + str(trial) +".txt")
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

mergedTrainingX = data['Bench Press']['t1']['accelerometer']['x'] + data['Bench Press']['t2']['accelerometer']['x'] + data['Bench Press']['t3']['accelerometer']['x'] + data['Bench Press']['t4']['accelerometer']['x']
mergedTrainingY = data['Bench Press']['t1']['accelerometer']['y'] + data['Bench Press']['t2']['accelerometer']['y'] + data['Bench Press']['t3']['accelerometer']['y'] + data['Bench Press']['t4']['accelerometer']['y']
mergedTrainingZ = data['Bench Press']['t1']['accelerometer']['z'] + data['Bench Press']['t2']['accelerometer']['z'] + data['Bench Press']['t3']['accelerometer']['z'] + data['Bench Press']['t4']['accelerometer']['z']

b,a  = signal.butter()