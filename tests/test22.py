#!/usr/bin/python
#coding:utf-8
# Test the AlhajyaseenAlgorithm
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
kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
al_algo = reference_implementations.AlhajyaseenAlgorithm()
for s in kitti_samples:
    al_algo.predict(s)
# al_algo = reference_implementations.AlhajyaseenAlgorithm()
# # coords = al_algo._get_euler_spiral_line(23.0,18.0,np.array([0,1]))
# plot_helper.plot_line('r',al_algo._get_curved_line(5.0, np.array([1,0.5]), np.array([0,1])))
# # x, y = zip(*coords)
# plt.axis('equal')
# # plt.plot(x,y)
# plt.show()
