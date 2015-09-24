#!/usr/bin/python
#coding:utf-8
# Test of prediction quality with different values for INT_DIST
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit",                       # Measured vehicle speed on exit way at INT_DIST
    "curvature_entry",
    "curvature_exit",
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]
sample_files = [
    "samples_10.pickle",
    "samples_20.pickle",
    "samples_30.pickle",
    "samples_40.pickle",
    "samples_50.pickle"
]

for fn in sample_files:
    rf_algo = regressors.RandomForestAlgorithm(feature_list)
    samples = automatic_test.load_samples('../data/training_data/samples_INT_DIST/'+fn)
    samples = automatic_test.normalize_features(samples)
    train_sample_sets, test_sample_sets = automatic_test.get_cross_validation_samples(samples, 0.8, 5)
    automatic_test.test([rf_algo], train_sample_sets, test_sample_sets, cross_validation=True)
# results = automatic_test.predict(algos, test_samples)
# automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case")
