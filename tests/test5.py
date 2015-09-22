#!/usr/bin/python
#coding:utf-8
# Comparing test of different features for lane distance with RandomForestRegressor
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations

base_feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit",                       # Measured vehicle speed on exit way at INT_DIST
    "curvature_entry",
    "curvature_exit"
]
feature_list_lane_exact = base_feature_list + [
    "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    "lane_distance_exit_exact"                  # Distance of track line to curve secant center point at 180 degree angle
]
feature_list_lane_center = base_feature_list + [
    "lane_distance_entry_lane_center",          # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_exit_lane_center"            # Distance of lane center line to curve secant ceter point at 180 degree angle
]
feature_list_lane_projected = base_feature_list + [
    "lane_distance_entry_projected_normal",     # Distance of track line to entry way at INT_DIST projected along normal
    "lane_distance_exit_projected_normal"       # Distance of track line to exit way at INT_DIST projected along normal
]
rf_algo_exact = regressors.RandomForestAlgorithm(feature_list_lane_exact)
rf_algo_center = regressors.RandomForestAlgorithm(feature_list_lane_center)
rf_algo_projected = regressors.RandomForestAlgorithm(feature_list_lane_projected)
algos = [rf_algo_exact, rf_algo_center, rf_algo_projected]
samples = automatic_test.load_samples('../data/training_data/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_sample_sets, test_sample_sets = automatic_test.get_cross_validation_samples(samples, 0.8, 5)
automatic_test.test(algos, train_sample_sets, test_sample_sets, cross_validation=True)
# results = automatic_test.predict(algos, test_samples)
# automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case")
