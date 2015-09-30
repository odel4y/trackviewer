#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression prediction from RandomForestExtendedAlgorithm
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types

feature_list = _feature_types

rf_algo = regressors.RandomForestExtendedAlgorithm(feature_list)
samples = automatic_test.load_samples('../data/training_data/samples_23_09_15/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_samples, test_samples = automatic_test.get_partitioned_samples(samples, 0.8)
automatic_test.test([rf_algo], train_samples, test_samples, cross_validation=False)
automatic_test.predict_all_estimators([rf_algo], test_samples)
