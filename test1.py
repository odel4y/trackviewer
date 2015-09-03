#!/usr/bin/python
#coding:utf-8
# Simple test with RandomForestRegressor and Automatic Testing
import automatic_test
import regressors
import reference_implementations

samples = automatic_test.load_samples('data/training_data/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    "lane_distance_exit_exact",                 # Distance of track line to curve secant center point at 180 degree angle
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit"                        # Measured vehicle speed on exit way at INT_DIST
]
rf_algo = regressors.RandomForestAlgorithm(feature_list)
ispline_algo = reference_implementations.InterpolatingSplineAlgorithm()
algos = [rf_algo, ispline_algo]
automatic_test.train(algos, train_samples)
automatic_test.test(algos, test_samples)
