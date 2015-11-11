#!/usr/bin/python
#coding:utf-8
# Show the feature importance by directly extracting it from the Random Forest
# Based on Scikit example (http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method
import numpy as np
import matplotlib.pyplot as plt

# By default test all available features
feature_list = _feature_types

rf_algo = regressors.RandomForestAlgorithm(feature_list)
kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08_rectified/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt_rectified/samples.pickle')
samples = kitti_samples + darmstadt_samples

select_label_method(samples, 'y_distances')
automatic_test.train([rf_algo], samples)
# Extract importances
importances = rf_algo.regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_algo.regressor.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(len(feature_list)):
    print("%d. %s (%f)" % (f + 1, _feature_types[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(feature_list)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(feature_list)), [_feature_types[i] for i in indices], rotation="vertical")
plt.xlim([-1, len(feature_list)])
plt.gcf().subplots_adjust(bottom=0.5)
plt.show()
