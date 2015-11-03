#!/usr/bin/python
#coding:utf-8
# Find out whether vehicle velocity is correctly calculated
from __future__ import division
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types
import extract_features
import numpy as np
import plot_helper
import matplotlib.pyplot as plt
import random
import pickle

def show_velocity_profile(int_sit):
    track = int_sit['track']
    velocities = []
    distance = []
    curr_distance = 0.

    for i in range(len(track)-1):
        this_p = np.array([track[i][0], track[i][1]])
        next_p = np.array([track[i+1][0], track[i+1][1]])
        this_dist = np.linalg.norm(next_p - this_p)
        time_diff = (track[i+1][2] - track[i][2]).total_seconds()
        curr_distance += this_dist
        velocities.append( this_dist/time_diff*3.6/1.5)
        distance.append(curr_distance)

    plt.plot(distance, velocities)
    plt.show()

files = ['2010_03_09_drive_0019_1.pickle','2010_03_09_drive_0019_2.pickle']
path = '../data/prepared_data/KITTI_and_Karlsruhe/'

kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
selected_samples = [s for s in kitti_samples if s['pickled_filename'].split('/')[-1] in files]

for s in selected_samples:
    automatic_test.output_sample_features(s)

for fn in files:
    fn = path + fn
    with open(fn, 'r') as f:
        int_sit = pickle.load(f)
        int_sit['track'] = extract_features.transform_track_to_cartesian(int_sit['track'])
        show_velocity_profile(int_sit)
