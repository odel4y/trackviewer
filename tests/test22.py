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
import numpy.random as random

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
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

kitti_samples = automatic_test.load_samples('../data/training_data/samples_kitti/samples_rectified.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_darmstadt/samples_rectified.pickle')
random.shuffle(kitti_samples)
random.shuffle(darmstadt_samples)
extract_features.select_label_method(kitti_samples, 'y_distances')
extract_features.select_label_method(darmstadt_samples, 'y_distances')
kitti_train_samples, test_samples = automatic_test.get_partitioned_samples(kitti_samples, 0.6)
train_samples = kitti_train_samples + darmstadt_samples

rf_algo = regressors.RandomForestAlgorithm(feature_list, single_target_variable=False)
al_algo = reference_implementations.AlhajyaseenAlgorithm(allow_rectification=False)
is_algo = reference_implementations.InterpolatingSplineAlgorithm()
algos = [rf_algo, al_algo, is_algo]
results = automatic_test.test(algos, train_samples, test_samples)
# automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case", orientation="curve-secant")
automatic_test.show_intersection_plot(results, test_samples, which_samples="all", orientation="curve-secant")
