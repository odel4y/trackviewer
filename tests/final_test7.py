#!/usr/bin/python
#coding:utf-8
# Test the AlhajyaseenAlgorithm
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types
from extract_features import *
import numpy as np
import plot_helper
import matplotlib.pyplot as plt
import numpy.random as random

# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.8)

random.seed(123456789)

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    # "maxspeed_exit",                            # Allowed maximum speed on exit way
    # "track_distance_along_curve_secant_entry",  # Distance of track line to curve secant center point at 0 degree angle
    # "track_distance_along_curve_secant_exit",   # Distance of track line to curve secant center point at 180 degree angle
    "track_distance_projected_along_half_angle_vec_entry",
    # "track_distance_projected_along_half_angle_vec_exit",
    "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
    # "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
    # "oneway_entry",                             # Is entry way a oneway street?
    # "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    # "curvature_exit",                           # Curvature of exit way over INT_DIST
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    # "vehicle_speed_exit",                       # Measured vehicle speed on exit way at INT_DIST
    # "lane_count_entry",                         # Total number of lanes in entry way
    # "lane_count_exit",                          # Total number of lanes in exit way
    # "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"
]

kitti_samples = automatic_test.load_samples('../data/training_data/samples_kitti/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_darmstadt/samples.pickle')
samples = kitti_samples + darmstadt_samples
select_label_method(samples, 'y_distances')
random.shuffle(samples)
sub_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
# train_samples, validation_samples = automatic_test.get_partitioned_samples(sub_samples, 0.75)
train_samples, validation_samples = automatic_test.get_cross_validation_samples(sub_samples, 4)

rf_algo = regressors.RandomForestAlgorithm(feature_list, single_target_variable=False, random_state=random, n_estimators=35, max_features=5, max_leaf_nodes=8)
al_algo = reference_implementations.AlhajyaseenAlgorithm(allow_rectification=True, des_exit_dist=0.0)
is_algo = reference_implementations.InterpolatingSplineAlgorithm()
algos = [rf_algo, al_algo, is_algo]
# algos = [rf_algo]
results = automatic_test.test(algos, train_samples, validation_samples, cross_validation=True, print_results=True)
# automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case", orientation="curve-secant")
# automatic_test.show_intersection_plot(results, validation_samples, which_samples="all", orientation="curve-secant")
