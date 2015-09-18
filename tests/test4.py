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
    # "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    # "lane_distance_exit_exact",                 # Distance of track line to curve secant center point at 180 degree angle
    "lane_distance_entry_lane_center",          # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_exit_lane_center",           # Distance of lane center line to curve secant ceter point at 180 degree angle
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit"                        # Measured vehicle speed on exit way at INT_DIST
]
rf_algo = regressors.RandomForestAlgorithm(feature_list)
rfc_algo = regressors.RFClassificationAlgorithm(feature_list, bin_num=20, min_radius=6.0, max_radius=28.0)
#ispline_algo = reference_implementations.InterpolatingSplineAlgorithm()
algos = [rf_algo, rfc_algo]
samples = automatic_test.load_samples('../data/training_data/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
automatic_test.train(algos, train_samples)
results = automatic_test.predict(algos, test_samples)
results_proba = automatic_test.predict_proba([rfc_algo], test_samples)
automatic_test.show_intersection_plot(results, test_samples, results_proba, which_samples="best-worst-case")
# automatic_test.show_graph_plot(results, test_samples, results_proba, which_samples="best-worst-case")
