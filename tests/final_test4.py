#!/usr/bin/python
#coding:utf-8
# Plot sample intersections with distances/radii and path prediction whilst extracting probability heatmap
import sys
sys.path.append('../')
import automatic_test
import regressors
from extract_features import _feature_types
from extract_features import *
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from plot_helper import plot_intersection, get_heatmap_from_polar_all_predictors, get_heatmap_from_distances_all_predictors

cmap = plt.get_cmap('Paired')

random.seed(123456789)

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

feature_list_with_path = [
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
sub_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
# train_sample_sets, validation_sample_sets = automatic_test.get_cross_validation_samples(sub_samples, 4)

random_state = random.get_state()

whole_algo = regressors.RandomForestAlgorithm(feature_list_with_path, random_state=random, n_estimators=35, max_features=5, max_leaf_nodes=8)
#
# select_label_method(sub_samples, 'y_radii')
# # train_samples, validation_samples = automatic_test.get_cross_validation_samples(sub_samples, 4)
# train_samples, validation_samples = automatic_test.get_partitioned_samples(sub_samples, 0.75)
# # Test 01 and 02
# automatic_test.train([whole_algo], train_samples)
# results = automatic_test.predict_all_estimators([whole_algo], validation_samples)
# for sample, prediction, predictions_all_estimators in zip(validation_samples, results[whole_algo]['predictions'], results[whole_algo]['predictions_all_estimators']):
#     automatic_test.output_sample_features(sample)
#     predicted_distances = [pred[0] for pred in predictions_all_estimators]
#     heatmap = get_heatmap_from_polar_all_predictors(predicted_distances, sample['geometry']['curve_secant'], sample['X'][_feature_types.index('intersection_angle')])
#     plot_intersection(sample, [prediction], rgbcolors=[cmap(0.6)], labels=[whole_algo.get_name()], heatmap=heatmap, orientation="curve-secant")


select_label_method(sub_samples, 'y_distances')
select_label_method(test_samples, 'y_distances')
# train_samples, validation_samples = automatic_test.get_cross_validation_samples(sub_samples, 4)
train_samples, validation_samples = automatic_test.get_partitioned_samples(sub_samples, 0.75)
# Test 03 and 04
automatic_test.train([whole_algo], train_samples)
# results = automatic_test.predict_all_estimators([whole_algo], test_samples)
results = automatic_test.predict_all_estimators([whole_algo], validation_samples)
for sample, prediction, predictions_all_estimators in zip(validation_samples, results[whole_algo]['predictions'], results[whole_algo]['predictions_all_estimators']):
# for sample, prediction, predictions_all_estimators in zip(test_samples, results[whole_algo]['predictions'], results[whole_algo]['predictions_all_estimators']):
    predicted_distances = [pred[0] for pred in predictions_all_estimators]
    half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
    heatmap = get_heatmap_from_distances_all_predictors(predicted_distances,
                                                    sample['geometry']['entry_line'],
                                                    sample['geometry']['exit_line'],
                                                    half_angle_vec)
    plot_intersection(sample, [prediction], rgbcolors=[cmap(0.6)], labels=[whole_algo.get_name()], heatmap=heatmap, orientation="curve-secant")
