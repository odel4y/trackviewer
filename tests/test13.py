#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression predictions from RandomForestAlgorithm and display each line separately
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types
from plot_helper import plot_intersection

feature_list = _feature_types

rf_algo = regressors.RandomForestAlgorithm(feature_list)
samples = automatic_test.load_samples('../data/training_data/samples_23_09_15/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
#automatic_test.test([rf_algo], train_samples, test_samples, cross_validation=False)
automatic_test.train([rf_algo], train_samples)
results = automatic_test.predict_all_estimators([rf_algo], test_samples)
for sample, predictions_all_estimators in zip(test_samples, results[rf_algo]['predictions_all_estimators']):
    predicted_radii = [pred[0] for pred in predictions_all_estimators]
    print predicted_radii
    plot_intersection(sample, predicted_radii, orientation="curve-secant")
