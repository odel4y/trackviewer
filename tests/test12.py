#!/usr/bin/python
#coding:utf-8
# Comparing random forest and Extra Trees algorithm
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types

feature_list = _feature_types

rf_algo = regressors.RandomForestAlgorithm(feature_list)
et_algo = regressors.ExtraTreesAlgorithm(feature_list)
algos = [rf_algo, et_algo]
samples = automatic_test.load_samples('../data/training_data/samples_23_09_15/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_sample_sets, test_sample_sets = automatic_test.get_cross_validation_samples(samples, 0.8, 5)
automatic_test.test(algos, train_sample_sets, test_sample_sets, cross_validation=True)
# results = automatic_test.predict(algos, test_samples)
# automatic_test.show_intersection_plot(results, test_samples, which_samples="best-worst-case")
