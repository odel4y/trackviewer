#!/usr/bin/python
#coding:utf-8
# Test RandomForestAlgorithm with different label methods
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method
from plot_helper import plot_intersection

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_lane_center",          # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_exit_lane_center",           # Distance of lane center line to curve secant ceter point at 180 degree angle
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

rf_algo_radii = regressors.RandomForestAlgorithm(feature_list)
rf_algo_distances = regressors.RandomForestAlgorithm(feature_list)
samples_radii = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples_radii = automatic_test.normalize_features(samples)
samples_distances = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples_distances = automatic_test.normalize_features(samples_distances)
select_label_method(samples_distances, 'y_distances')
train_samples_radii, test_samples_radii = automatic_test.get_cross_validation_samples(samples_radii, 0.7, 5)
train_samples_distances, test_samples_distances = automatic_test.get_cross_validation_samples(samples_distances, 0.7, 5)
automatic_test.test([rf_algo_radii], train_samples_radii, test_samples_radii, cross_validation=True)
automatic_test.test([rf_algo_distances], train_samples_distances, test_samples_distances, cross_validation=True)
# automatic_test.train([rf_algo_distances], train_samples_distances)
# results = automatic_test.predict([rf_algo_distances], test_samples_distances)
# automatic_test.show_intersection_plot(results, test_samples_distances, which_samples="all")
