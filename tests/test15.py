#!/usr/bin/python
#coding:utf-8
# Show all samples together with feature vector
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method
from plot_helper import plot_intersection

feature_list = _feature_types

rf_algo = regressors.RandomForestAlgorithm(feature_list)
samples = automatic_test.load_samples('../data/training_data/samples_kitti/samples.pickle')
# samples = automatic_test.normalize_features(samples)
select_label_method(samples, 'y_distances')
for sample in samples:
    fn = sample['pickled_filename'].split('/')[-1]
    # Print all the feature values
    print "=====", fn
    for f in feature_list:
        print f, ":", sample['X'][_feature_types.index(f)]
    plot_intersection(sample, title=fn, orientation="curve-secant")
