#!/usr/bin/python
#coding:utf-8
# Show the curvatures of right turn and left turn samples respectively
import sys
sys.path.append('../')
import automatic_test
import extract_features
from extract_features import _feature_types
import plot_helper
import sample_analysis

samples = automatic_test.load_samples('../data/training_data/samples.pickle')
right_turn_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] <= 0.0]
left_turn_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] > 0.0]
sample_analysis.plot_sample_intersection_curvature(right_turn_samples, "Right turn sample curvature over intersection coordinates")
sample_analysis.plot_sample_intersection_curvature(left_turn_samples, "Left turn sample curvature over intersection coordinates")
