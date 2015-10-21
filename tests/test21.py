#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression predictions from RandomForestAlgorithm and display each line separately
# Use Darmstadt and KITTI samples
# Label method: y_distances
import sys
sys.path.append('../')
import automatic_test
import regressors
import reference_implementations
from extract_features import _feature_types, select_label_method, get_half_angle_vec
from plot_helper import plot_intersection, get_heatmap_from_distances_all_predictors

# feature_list = _feature_types
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

rf_algo = regressors.RandomForestAlgorithm(feature_list)
is_algo = reference_implementations.InterpolatingSplineAlgorithm()
kitti_samples = automatic_test.load_samples('../data/training_data/samples_15_10_12_rectified/samples.pickle')
darmstadt_samples = automatic_test.load_samples('../data/training_data/samples_15_10_20_darmstadt_rectified/samples.pickle')
select_label_method(kitti_samples, 'y_distances')
select_label_method(darmstadt_samples, 'y_distances')
train_samples_kitti, test_samples_kitti = automatic_test.get_partitioned_samples(kitti_samples, 0.7)
train_samples_darmstadt, test_samples_darmstadt = automatic_test.get_partitioned_samples(darmstadt_samples, 0.7)
train_samples = train_samples_kitti + train_samples_darmstadt
test_samples = test_samples_darmstadt + test_samples_kitti
automatic_test.train([rf_algo], train_samples)
results = automatic_test.predict_all_estimators([rf_algo], test_samples)
is_results = automatic_test.predict([is_algo], test_samples)

for sample, rf_prediction, rf_predictions_all_estimators, is_prediction in zip(test_samples,
                                                            results[rf_algo]['predictions'],
                                                            results[rf_algo]['predictions_all_estimators'],
                                                            is_results[is_algo]['predictions']):
    predicted_distances = [pred[0] for pred in rf_predictions_all_estimators]
    half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
    heatmap = get_heatmap_from_distances_all_predictors(predicted_distances,
                                                    sample['geometry']['entry_line'],
                                                    sample['geometry']['exit_line'],
                                                    half_angle_vec)
    # plot_intersection(sample, predicted_distances, heatmap=heatmap, orientation="curve-secant")
    automatic_test.output_sample_features(sample, feature_list)
    plot_intersection(sample, [rf_prediction, is_prediction], rgbcolors=['b', 'g'], labels=['RF Algorithm', 'Spline Algorithm'], heatmap=heatmap, orientation="curve-secant")
