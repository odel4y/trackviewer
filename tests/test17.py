#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression predictions from RandomForestAlgorithm and display each line separately
# Label method: y_distances
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method, get_half_angle_vec
import extract_features
from plot_helper import plot_intersection, get_heatmap_from_distances_all_predictors, get_heatmap_from_polar_all_predictors
import numpy.random as random

feature_list = _feature_types

rf_algo = regressors.RandomForestAlgorithm(feature_list)
kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt/samples.pickle')
extract_features.select_label_method(kitti_samples, 'y_radii')
extract_features.select_label_method(darmstadt_samples, 'y_radii')
random.shuffle(kitti_samples)
random.shuffle(darmstadt_samples)
kitti_train_samples, kitti_test_samples = automatic_test.get_partitioned_samples(kitti_samples, 0.7)
darmstadt_train_samples, darmstadt_test_samples = automatic_test.get_partitioned_samples(darmstadt_samples, 0.7)
train_samples = kitti_train_samples + darmstadt_train_samples
test_samples = kitti_test_samples + darmstadt_test_samples

# samples = automatic_test.load_samples('../data/training_data/samples.pickle')
# # samples = automatic_test.normalize_features(samples)
# select_label_method(samples, 'y_distances')
# train_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
# #automatic_test.test([rf_algo], train_samples, test_samples, cross_validation=False)
automatic_test.train([rf_algo], train_samples)
results = automatic_test.predict_all_estimators([rf_algo], test_samples)

for sample, prediction, predictions_all_estimators in zip(test_samples, results[rf_algo]['predictions'], results[rf_algo]['predictions_all_estimators']):
    predicted_distances = [pred[0] for pred in predictions_all_estimators]


    heatmap = get_heatmap_from_polar_all_predictors(predicted_distances, sample['geometry']['curve_secant'], sample['X'][_feature_types.index('intersection_angle')])

    # half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
    # heatmap = get_heatmap_from_distances_all_predictors(predicted_distances,
    #                                                 sample['geometry']['entry_line'],
    #                                                 sample['geometry']['exit_line'],
    #                                                 half_angle_vec)
    # plot_intersection(sample, predicted_distances, heatmap=heatmap, orientation="curve-secant")
    plot_intersection(sample, [prediction], rgbcolors=['b'], heatmap=heatmap, orientation="curve-secant")
