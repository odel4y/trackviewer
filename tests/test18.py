#!/usr/bin/python
#coding:utf-8
# Extract single estimator regression predictions from RandomForestAlgorithm and display each line separately
# Label method: y_distances
# Test all samples and visually check for irregularities
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
    "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    "lane_distance_entry_lane_center",          # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_exit_lane_center",           # Distance of lane center line to curve secant ceter point at 180 degree angle
    "lane_distance_entry_projected_normal",     # Distance of track line to entry way at INT_DIST projected along normal
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "bicycle_designated_entry",                 # Is there a designated bicycle way in the entry street?
    "bicycle_designated_exit",                  # Is there a designated bicycle way in the exit street?
    "lane_count_entry",                         # Total number of lanes in entry way
    "lane_count_exit",                          # Total number of lanes in exit way
    "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]

rf_algo_distances = regressors.RandomForestAlgorithm(feature_list)
samples_distances = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples = automatic_test.normalize_features(samples)
select_label_method(samples_distances, 'y_distances')
train_sample_sets_distances, test_sample_sets_distances = automatic_test.get_cross_validation_samples(samples_distances, 0.8, 5)

rf_algo_radii = regressors.RandomForestAlgorithm(feature_list)
samples_radii = automatic_test.load_samples('../data/training_data/samples.pickle')
# samples = automatic_test.normalize_features(samples)
select_label_method(samples_radii, 'y_radii')
train_sample_sets_radii, test_sample_sets_radii = automatic_test.get_cross_validation_samples(samples_radii, 0.8, 5)

for train_samples_distances, test_samples_distances, train_samples_radii, test_samples_radii in zip(train_sample_sets_distances, test_sample_sets_distances, train_sample_sets_radii, test_sample_sets_radii):
    automatic_test.train([rf_algo_distances], train_samples_distances)
    results_distances = automatic_test.predict_all_estimators([rf_algo_distances], test_samples_distances)

    automatic_test.train([rf_algo_radii], train_samples_radii)
    results_radii = automatic_test.predict_all_estimators([rf_algo_radii], test_samples_radii)

    for sample, prediction, predictions_all_estimators, prediction_radii in zip(test_samples_distances, results_distances[rf_algo_distances]['predictions'], results_distances[rf_algo_distances]['predictions_all_estimators'], results_radii[rf_algo_radii]['predictions']):
        automatic_test.output_sample_features(sample, rf_algo_distances.features)
        predicted_distances = [pred[0] for pred in predictions_all_estimators]
        half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
        heatmap = get_heatmap_from_distances_all_predictors(predicted_distances,
                                                        sample['geometry']['entry_line'],
                                                        sample['geometry']['exit_line'],
                                                        half_angle_vec)
        # plot_intersection(sample, predicted_distances, heatmap=heatmap, orientation="curve-secant")
        plot_intersection(  sample,
                            [prediction, prediction_radii],
                            rgbcolors=['b', 'c'],
                            label_methods=['y_distances', 'y_radii'],
                            labels=['distances', 'radii'],
                            title=sample['pickled_filename'],
                            heatmap=heatmap,
                            orientation="curve-secant")
