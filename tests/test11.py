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
# feature_list = _feature_types

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_along_curve_secant_entry",   # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_along_curve_secant_exit",    # Distance of lane center line to curve secant ceter point at 180 degree angle
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "bicycle_designated_entry",                 # Is there a designated bicycle way in the entry street?
    "bicycle_designated_exit",                  # Is there a designated bicycle way in the exit street?
    "lane_count_entry",                         # Total number of lanes in entry way
    "lane_count_exit",                          # Total number of lanes in exit way
    "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]

rf_algo = regressors.RandomForestAlgorithm(feature_list)
kitti_samples = automatic_test.load_samples('../data/training_data/samples_kitti/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_darmstadt/samples.pickle')
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
    print("%d. %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(feature_list)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(feature_list)), [feature_list[i] for i in indices], rotation="vertical")
plt.xlim([-1, len(feature_list)])
plt.gcf().subplots_adjust(bottom=0.5)
plt.show()
