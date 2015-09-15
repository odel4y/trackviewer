#!/usr/bin/python
#coding:utf-8
# Systematic test of feature quality with RandomForestRegressor
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    "lane_distance_exit_exact",                 # Distance of track line to curve secant center point at 180 degree angle
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit"                        # Measured vehicle speed on exit way at INT_DIST
]
samples = automatic_test.load_samples('../data/training_data/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_sample_sets, test_sample_sets = automatic_test.get_cross_validation_samples(samples, 0.7, 5)
automatic_test.test_feature_permutations(
                    regressors.RandomForestAlgorithm,
                    train_sample_sets,
                    test_sample_sets,
                    feature_list,
                    min_num_features=6,
                    cross_validation=True)
