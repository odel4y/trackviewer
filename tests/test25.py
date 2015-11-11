#!/usr/bin/python
#coding:utf-8
# Testing out different parameters of RandomForestRegressor and their performance
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
import numpy.random as random

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_projected_normal",
    "lane_distance_exit_projected_normal",
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "bicycle_designated_entry",                 # Is there a designated bicycle way in the entry street?
    "bicycle_designated_exit",                  # Is there a designated bicycle way in the exit street?
    "lane_count_entry",                         # Total number of lanes in entry way
    "lane_count_exit",                          # Total number of lanes in exit way
    "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]

random.seed(42)

kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08_rectified/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt_rectified/samples.pickle')
extract_features.select_label_method(kitti_samples, 'y_distances')
extract_features.select_label_method(darmstadt_samples, 'y_distances')
random.shuffle(kitti_samples)
random.shuffle(darmstadt_samples)
kitti_train_samples, kitti_test_samples = automatic_test.get_partitioned_samples(kitti_samples, 0.7)
darmstadt_train_samples, darmstadt_test_samples = automatic_test.get_partitioned_samples(darmstadt_samples, 0.7)
train_samples = kitti_train_samples + darmstadt_train_samples
test_samples = kitti_test_samples + darmstadt_test_samples

parameter = "max_depth"
values = np.arange(1,40,2)
runs = 10
params_mse = automatic_test.test_parameter_variations(regressors.RandomForestAlgorithm, {"features":feature_list, "random_state":random}, parameter, values, train_samples, test_samples, runs)
plt.plot(values, params_mse, 'b.-')
plt.xlabel(parameter)
plt.show()

# rf_algo = regressors.RandomForestAlgorithm(feature_list, single_target_variable=False, n_estimators=80)
# algos = [rf_algo]
# results = automatic_test.test(algos, train_samples, test_samples)
# # automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case", orientation="curve-secant")
# automatic_test.show_intersection_plot(results, test_samples, which_samples="worst-case", orientation="curve-secant")
