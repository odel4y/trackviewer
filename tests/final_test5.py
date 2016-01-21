#!/usr/bin/python
#coding:utf-8
# Plot the learning rate of the algorithm with different train, test sets
from __future__ import division
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
import seaborn as sns
import random
import pickle

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.8)

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
select_label_method(samples, 'y_distances')
random.shuffle(samples)
sub_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
# train_sample_sets, validation_sample_sets = automatic_test.get_cross_validation_samples(sub_samples, 5)
# rf_algo = regressors.RandomForestAlgorithm(feature_list)
rf_algo = regressors.RandomForestAlgorithm(feature_list, single_target_variable=False, n_estimators=35, max_features=min(5,len(feature_list)), max_leaf_nodes=8)

N_training_samples = 89
N_cv_samples = len(sub_samples) - N_training_samples

N_tests = 10
training_samples_steps = np.arange(10, N_training_samples+1, 5)
all_training_errors = []
all_cv_errors = []
all_training_samples_sizes = []

for test_i in range(N_tests):
    print "====== Test number %d ======" % test_i
    random.shuffle(sub_samples)
    cv_samples = sub_samples[N_training_samples:]
    for training_samples_size in training_samples_steps:
        print "Training samples size:", training_samples_size
        training_samples = sub_samples[:training_samples_size]

        automatic_test.train([rf_algo], training_samples)
        rs_train = automatic_test.get_result_statistics(automatic_test.predict([rf_algo], training_samples))
        rs_cv = automatic_test.get_result_statistics(automatic_test.predict([rf_algo], cv_samples))

        all_training_samples_sizes.append(training_samples_size)
        all_training_errors.append(rs_train[rf_algo]['mean_mse'])
        all_cv_errors.append(rs_cv[rf_algo]['mean_mse'])

training_errors = []
cv_errors = []
training_samples_sizes = []

# Flatten the results
for training_samples_size in training_samples_steps:
    indices = [i for i,x in enumerate(all_training_samples_sizes) if x == training_samples_size]

    training_samples_sizes.append(training_samples_size)
    total_t_error = 0.
    total_cv_error = 0.
    for i in indices:
        total_t_error += all_training_errors[i]
        total_cv_error += all_cv_errors[i]
    training_errors.append(total_t_error/len(indices))
    cv_errors.append(total_cv_error/len(indices))

with open('learning_rate_result.pickle', 'w') as f:
    pickle.dump((training_samples_sizes, training_errors, cv_errors), f)

fig = plt.figure()
plt.hold(True)
te_line, = sns.plt.plot(training_samples_sizes, training_errors, 'r.-', linewidth=2.0)
cve_line, = sns.plt.plot(training_samples_sizes, cv_errors, 'b.-', linewidth=2.0)
sns.plt.gca().set_ylim([1, 10.0])
plt.legend([te_line, cve_line], ['Training Error', 'Cross Validation Error'])
plt.show()
