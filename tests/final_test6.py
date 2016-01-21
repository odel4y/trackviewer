#!/usr/bin/python
#coding:utf-8
# Hyperparameter optimization with Random Search with penalization for overfitting
import sys
sys.path.append('../')
import automatic_test
import regressors
from extract_features import _feature_types
from extract_features import *
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

random.seed(123456789)

# feature_list = [
#     "intersection_angle",                       # Angle between entry and exit way
#     "maxspeed_entry",                           # Allowed maximum speed on entry way
#     "maxspeed_exit",                            # Allowed maximum speed on exit way
#     "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
#     "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
#     "oneway_entry",                             # Is entry way a oneway street?
#     "oneway_exit",                              # Is exit way a oneway street?
#     "curvature_entry",                          # Curvature of entry way over INT_DIST
#     "curvature_exit",                           # Curvature of exit way over INT_DIST
#     "lane_count_entry",                         # Total number of lanes in entry way
#     "lane_count_exit",                          # Total number of lanes in exit way
#     "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
#     "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
# ]

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
random.shuffle(samples)
select_label_method(samples, 'y_distances')
sub_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
train_sample_sets, validation_sample_sets = automatic_test.get_cross_validation_samples(sub_samples, 4)

random_state = random.get_state()

algo_args = {
    'features': feature_list,
    'single_target_variable': False,
    'n_jobs': 1
}
hyp_intervals = [
    ('n_estimators', 1, 200),
    ('max_leaf_nodes', 5, len(train_sample_sets[0])),
    ('max_features', 1, len(feature_list))
]

search_results = automatic_test.random_search_hyperparameters(
                                    regressors.RandomForestAlgorithm,
                                    algo_args,
                                    random_state,
                                    train_sample_sets,
                                    validation_sample_sets,
                                    hyp_intervals,
                                    120,
                                    calc_train_error=True)

cost_function = lambda r: r[2]['mean_mse']+1.0*np.power(abs(r[2]['mean_mse']-r[3]['mean_mse']),2)

sorted_search_results = sorted(search_results, key=cost_function)
# Display the 10 best solutions
for position_i, (try_i, hyp_values, rs, trs) in enumerate(sorted_search_results[:10]):
    print "#%d (Try %d) MSE: %.2f (%.2f) CVE-TE: %.2f" % (position_i, try_i, rs['mean_mse'], rs['std_mse'], abs(rs['mean_mse']-trs['mean_mse']))
    for param_name, param_value in sorted(hyp_values.items()):
        print "%s: %d" % (param_name, param_value)
