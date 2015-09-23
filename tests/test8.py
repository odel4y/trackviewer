#!/usr/bin/python
#coding:utf-8
# Test with different numbers of estimators in RandomForestRegressor
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
mse_mean = np.array([])
mse_std = np.array([])
n_est_hist = np.array([])
samples = automatic_test.load_samples('../data/training_data/samples.pickle')
samples = automatic_test.normalize_features(samples)
train_sample_sets, test_sample_sets = automatic_test.get_cross_validation_samples(samples, 0.8, 5)
for n_est in range(1,40, 2):
    rf_algo = regressors.RandomForestAlgorithm(feature_list, n_estimators = n_est)
    results = automatic_test.test([rf_algo], train_sample_sets, test_sample_sets, cross_validation=True)
    result_statistics = automatic_test.get_result_statistics(results)
    mse_mean = np.append(mse_mean, result_statistics[rf_algo]['average_mse'])
    mse_std = np.append(mse_std, result_statistics[rf_algo]['std_mse'])
    n_est_hist = np.append(n_est_hist, n_est)
# Plot the results
plt.hold(True)
plt.plot(n_est_hist, mse_mean)
plt.plot(n_est_hist, mse_mean + mse_std, 'r-')
plt.plot(n_est_hist, mse_mean - mse_std, 'r-')
plt.xlabel('n_estimators')
plt.ylabel('average_mse')
plt.show()
