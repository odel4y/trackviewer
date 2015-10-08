#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression predictions from RandomForestAlgorithm and display each line separately
# Label method: y_distances
# Test all samples and visually check for irregularities
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method, get_half_angle_vec
from plot_helper import plot_intersection, get_heatmap_from_distances_all_predictors

feature_list = _feature_types

rf_algo_distances = regressors.RandomForestAlgorithm(feature_list)
samples_distances = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples = automatic_test.normalize_features(samples)
select_label_method(samples_distances, 'y_distances')
train_sample_sets_distances, test_sample_sets_distances = automatic_test.get_cross_validation_samples(samples_distances, 0.8, 5)

rf_algo_radii = regressors.RandomForestAlgorithm(feature_list)
samples_radii = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples = automatic_test.normalize_features(samples)
select_label_method(samples_radii, 'y_radii')
train_sample_sets_radii, test_sample_sets_radii = automatic_test.get_cross_validation_samples(samples_radii, 0.8, 5)

for train_samples_distances, test_samples_distances, train_samples_radii, test_samples_radii in zip(train_sample_sets_distances, test_sample_sets_distances, train_sample_sets_radii, test_sample_sets_radii):
    automatic_test.train([rf_algo_distances], train_samples_distances)
    results_distances = automatic_test.predict_all_estimators([rf_algo_distances], test_samples_distances)

    automatic_test.train([rf_algo_radii], train_samples_radii)
    results_radii = automatic_test.predict_all_estimators([rf_algo_radii], test_samples_radii)

    for sample, prediction, predictions_all_estimators, prediction_radii in zip(test_samples_distances, results_distances[rf_algo_distances]['predictions'], results_distances[rf_algo_distances]['predictions_all_estimators'], results_radii[rf_algo_radii]['predictions']):
        automatic_test.output_sample_features(sample, rf_algo_distances.features)
        predicted_distances = [pred[0] for pred in predictions_all_estimators]
        half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
        heatmap = get_heatmap_from_distances_all_predictors(predicted_distances,
                                                        sample['geometry']['entry_line'],
                                                        sample['geometry']['exit_line'],
                                                        half_angle_vec)
        # plot_intersection(sample, predicted_distances, heatmap=heatmap, orientation="curve-secant")
        plot_intersection(  sample,
                            [prediction, prediction_radii],
                            rgbcolors=['b', 'c'],
                            label_methods=['y_distances', 'y_radii'],
                            title=sample['pickled_filename'],
                            heatmap=heatmap,
                            orientation="curve-secant")
