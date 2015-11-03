#!/usr/bin/python
#coding:utf-8
# Testing algorithm performance with and without darmstadt samples
# Also with and without rectified samples (Many errors are caused by incorrect OSM mapping)
import sys
sys.path.append('../')
import automatic_test
import extract_features
import regressors

feature_list = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_projected_normal",
    "lane_distance_exit_projected_normal",
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

kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_08/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt/samples.pickle')
extract_features.select_label_method(kitti_samples, 'y_distances')
extract_features.select_label_method(darmstadt_samples, 'y_distances')
train_kitti, test_kitti = automatic_test.get_partitioned_samples(kitti_samples, 0.7)
train_darmstadt, test_darmstadt = automatic_test.get_partitioned_samples(darmstadt_samples, 0.7)
train_samples = train_kitti + train_darmstadt
test_samples = test_kitti + test_darmstadt
rf_algo = regressors.RandomForestAlgorithm(feature_list)

print "###### Non-rectified data ######"
print "------ Only KITTI ------"
automatic_test.test([rf_algo], train_kitti, test_kitti, cross_validation=False)
print "------ KITTI + Darmstadt ------"
automatic_test.test([rf_algo], train_samples, test_samples, cross_validation=False)

print "###### Rectified data ######"
kitti_samples_rectified = automatic_test.load_samples('../data/training_data/samples_15_10_08_rectified/samples.pickle')
darmstadt_samples_rectified = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt_rectified/samples.pickle')
extract_features.select_label_method(kitti_samples_rectified, 'y_distances')
extract_features.select_label_method(darmstadt_samples_rectified, 'y_distances')
train_kitti_rectified, test_kitti_rectified = automatic_test.get_partitioned_samples(kitti_samples_rectified, 0.7)
train_darmstadt_rectified, test_darmstadt_rectified = automatic_test.get_partitioned_samples(darmstadt_samples_rectified, 0.7)
train_samples_rectified = train_kitti_rectified + train_darmstadt_rectified
test_samples_rectified = test_kitti_rectified + test_darmstadt_rectified
print "------ Only KITTI ------"
automatic_test.test([rf_algo], train_kitti_rectified, test_kitti_rectified, cross_validation=False)
print "------ KITTI + Darmstadt ------"
automatic_test.test([rf_algo], train_samples_rectified, test_samples_rectified, cross_validation=False)
