#!/usr/bin/python
#coding:utf-8
# Test a regressor trained with KITTI data on CMU data
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
import numpy as np
import matplotlib.pyplot as plt

feature_list = [
    "lane_distance_entry_exact",
    "curve_secant_dist",
    "lane_distance_exit_exact",
    "maxspeed_entry",
    "vehicle_speed_entry",
    "vehicle_speed_exit",
    "curvature_exit",
    "oneway_exit",
    "lane_count_entry",
    "has_right_of_way",
    "curvature_entry"
]

kitti_samples = automatic_test.load_samples('../data/training_data/samples_23_09_15/samples.pickle')
kitti_samples = automatic_test.normalize_features(kitti_samples)
cmu_samples = automatic_test.load_samples('../data/training_data/samples_CMU/samples.pickle')
cmu_samples = automatic_test.normalize_features(cmu_samples)
kitti_train_samples, kitti_test_samples = automatic_test.get_partitioned_samples(kitti_samples, 0.8)

rf_algo = regressors.RandomForestAlgorithm(feature_list)
automatic_test.test([rf_algo], kitti_train_samples, kitti_test_samples, cross_validation=False)
automatic_test.test([rf_algo], kitti_train_samples, cmu_samples, cross_validation=False)
