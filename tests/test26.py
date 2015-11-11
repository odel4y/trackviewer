#!/usr/bin/python
#coding:utf-8
# Plot prediction error in correlation to intersection angle
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

kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt/samples.pickle')
extract_features.select_label_method(kitti_samples, 'y_distances')
extract_features.select_label_method(darmstadt_samples, 'y_distances')
random.shuffle(kitti_samples)
random.shuffle(darmstadt_samples)
kitti_train_samples, kitti_test_samples = automatic_test.get_partitioned_samples(kitti_samples, 0.7)
darmstadt_train_samples, darmstadt_test_samples = automatic_test.get_partitioned_samples(darmstadt_samples, 0.7)
train_samples = kitti_train_samples + darmstadt_train_samples
test_samples = kitti_test_samples + darmstadt_test_samples

rf_algo = regressors.RandomForestAlgorithm(_feature_types)

results = automatic_test.test([rf_algo], train_samples, test_samples)

angles = np.rad2deg( np.array([s['X'][_feature_types.index('intersection_angle')] for s in test_samples]) )
plt.plot(angles, results[rf_algo]['mse'], 'r.')
plt.show(block=False)
automatic_test.show_intersection_plot(results, test_samples, which_samples="worst-case", orientation="curve-secant")
