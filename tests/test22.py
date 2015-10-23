#!/usr/bin/python
#coding:utf-8
# Test the AlhajyaseenAlgorithm
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types
import extract_features

kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
al_algo = reference_implementations.AlhajyaseenAlgorithm()
for s in kitti_samples:
    al_algo.predict(s)
