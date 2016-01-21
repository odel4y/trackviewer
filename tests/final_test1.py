#!/usr/bin/python
#coding:utf-8
# Show the feature importance by directly extracting it from the Random Forest
# Based on Scikit example (http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
# Show importance with and without path features
import sys
sys.path.append('../')
import automatic_test
import regressors
from extract_features import _feature_types
from extract_features import *
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.8)
sns.plt.rc("figure", figsize=[9,3])

def print_importances(rf_algo):
    # Extract importances
    importances = rf_algo.regressor.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_algo.regressor.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(feature_list)):
        print("%d. %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    sns.plt.figure()
    # sns.plt.title("Feature importances")
    sns.plt.bar(range(len(feature_list)), importances[indices],
           color="r", yerr=std[indices], align="center")
    sns.plt.xticks(range(len(feature_list)), ["%d."%(i+1) for i in range(len(feature_list))])
    # sns.plt.xticks(range(len(feature_list)), [feature_list[i] for i in indices], rotation="vertical")
    # plt.tick_params(
    # labelbottom='off') # labels along the bottom edge are off
    sns.plt.xlim([-1, len(feature_list)])
    # sns.plt.gcf().subplots_adjust(bottom=0.5)
    sns.plt.show()

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
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "lane_count_entry",                         # Total number of lanes in entry way
    "lane_count_exit",                          # Total number of lanes in exit way
    "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]

kitti_samples = automatic_test.load_samples('../data/training_data/samples_kitti/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_darmstadt/samples.pickle')
samples = kitti_samples + darmstadt_samples
random.shuffle(samples)
sub_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)

random_state = random.get_state()
#
# random.set_state(random_state)
# select_label_method(samples, 'y_radii')
# rf_algo = regressors.RandomForestAlgorithm(feature_list, random_state=random, n_estimators=150, max_leaf_nodes=10)
# automatic_test.train([rf_algo], samples)
# print_importances(rf_algo)
#
# random.set_state(random_state)
# select_label_method(sub_samples, 'y_distances')
# rf_algo = regressors.RandomForestAlgorithm(feature_list, random_state=random, n_estimators=27, max_features=12, max_leaf_nodes=5)
# automatic_test.train([rf_algo], sub_samples)
# print_importances(rf_algo)

# feature_list = [
#     "intersection_angle",                       # Angle between entry and exit way
#     "maxspeed_entry",                           # Allowed maximum speed on entry way
#     # "maxspeed_exit",                            # Allowed maximum speed on exit way
#     # "track_distance_along_curve_secant_entry",  # Distance of track line to curve secant center point at 0 degree angle
#     # "track_distance_along_curve_secant_exit",   # Distance of track line to curve secant center point at 180 degree angle
#     "track_distance_projected_along_half_angle_vec_entry",
#     # "track_distance_projected_along_half_angle_vec_exit",
#     "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
#     # "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
#     # "oneway_entry",                             # Is entry way a oneway street?
#     # "oneway_exit",                              # Is exit way a oneway street?
#     "curvature_entry",                          # Curvature of entry way over INT_DIST
#     "curvature_exit",                           # Curvature of exit way over INT_DIST
#     "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
#     # "vehicle_speed_exit",                       # Measured vehicle speed on exit way at INT_DIST
#     # "lane_count_entry",                         # Total number of lanes in entry way
#     "lane_count_exit",                          # Total number of lanes in exit way
#     # "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
#     "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
# ]
#
# random.set_state(random_state)
# select_label_method(sub_samples, 'y_radii')
# rf_algo = regressors.RandomForestAlgorithm(feature_list, random_state=random)
# automatic_test.train([rf_algo], sub_samples)
# print_importances(rf_algo)

random.set_state(random_state)
select_label_method(sub_samples, 'y_radii')
rf_algo = regressors.RandomForestAlgorithm(feature_list, random_state=random, n_estimators=27, max_features=min(12, len(feature_list)), max_leaf_nodes=5)
automatic_test.train([rf_algo], sub_samples)
print_importances(rf_algo)
