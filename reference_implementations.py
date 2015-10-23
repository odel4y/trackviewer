#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from extract_features import extended_interpolate, get_normal_to_line, \
            sample_line, _feature_types, get_offset_point_at_distance, extend_line, \
            sample_line_all
from constants import LANE_WIDTH, INT_DIST
import automatic_test

def parametric_combined_spline(x, y, k=3, resolution=100, kv=None, s=None):
    """Return a linear combination of parametric univariate splines through x, y evaluated at resolution"""
    x = np.array(x)
    y = np.array(y)

    nt = np.linspace(0, 1, resolution)

    # Prepare linear combination of splines with given knot vector
    tckp,u = scipy.interpolate.splprep([x,y],k=(k or 3),t=kv,s=s)
    x2, y2 = scipy.interpolate.splev(np.linspace(0,1,400), tckp)

    return x2, y2

class GeigerAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Quadratic B-Spline (Geiger 2014)'

    def predict(self, sample):
        predicted_line = self.get_geiger_line(sample['geometry']['entry_line'], sample['geometry']['exit_line'])
        radii = sample_line(sample['geometry']['curve_secant'],
                            predicted_line,
                            sample['X'][_feature_types.index('intersection_angle')])
        return radii

    def get_geiger_line(self, entry_line, exit_line):
        # w = 2 * LANE_WIDTH
        w = 30.0
        center_p = exit_line.interpolate(0.)
        far_entry_p =   get_offset_point_at_distance(entry_line, entry_line.length - 70.0, LANE_WIDTH/2)
        entry_p =       get_offset_point_at_distance(entry_line, entry_line.length - w, LANE_WIDTH/2)
        far_exit_p =    get_offset_point_at_distance(exit_line, 70.0, LANE_WIDTH/2)
        exit_p =        get_offset_point_at_distance(exit_line, w, LANE_WIDTH/2)
        coords = [  list(far_entry_p.coords)[0],
                    list(entry_p.coords)[0],
                    list(center_p.coords)[0],
                    list(exit_p.coords)[0],
                    list(far_exit_p.coords)[0]]
        x, y = zip(*coords)
        # Make a parametric quadratic spline with given knot vector
        x2, y2 = parametric_combined_spline(x, y, k=2, kv=[0., 0., 0., 0.1, 0.9, 1., 1., 1.])
        geiger_path_line = LineString(zip(x2, y2))
        return geiger_path_line

class ModifiedGeigerAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Quadratic B-Spline (Modified Geiger 2014)'

    def predict(self, sample):
        predicted_line = self.get_modified_geiger_line(sample['geometry']['entry_line'], sample['geometry']['exit_line'])
        radii = sample_line(sample['geometry']['curve_secant'],
                            predicted_line,
                            sample['X'][_feature_types.index('intersection_angle')])
        return radii

    def get_modified_geiger_line(self, entry_line, exit_line):
        # w = 2 * LANE_WIDTH
        w = 30.0
        lane_center_line_entry = extend_line(entry_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="forward")
        lane_center_line_exit = extend_line(exit_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="backward")
        center_p = lane_center_line_entry.intersection(lane_center_line_exit)
        print center_p
        center_p_dist_entry = lane_center_line_entry.project(center_p)
        center_p_dist_exit = lane_center_line_exit.project(center_p)
        far_entry_p = lane_center_line_entry.interpolate(center_p_dist_entry - 70.0)
        entry_p = lane_center_line_entry.interpolate(center_p_dist_entry - w)
        exit_p = lane_center_line_exit.interpolate(center_p_dist_exit + w)
        far_exit_p = lane_center_line_exit.interpolate(center_p_dist_exit + 70.0)
        coords = [  list(far_entry_p.coords)[0],
                    list(entry_p.coords)[0],
                    list(center_p.coords)[0],
                    list(exit_p.coords)[0],
                    list(far_exit_p.coords)[0]]
        x, y = zip(*coords)
        # Make a parametric quadratic spline with given knot vector
        x2, y2 = parametric_combined_spline(x, y, k=2, kv=[0., 0., 0., 0.1, 0.9, 1., 1., 1.])
        return LineString(zip(x2, y2))

class InterpolatingSplineAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Standard Interpolating Spline (k=2)'

    def predict(self, sample):
        predicted_line = self.get_interpolating_spline_line(sample['geometry']['entry_line'], sample['geometry']['exit_line'])
        pred = sample_line_all(predicted_line,
                            sample['label']['selected_method'],
                            sample)
        return pred

    def get_interpolating_spline_line(self, entry_line, exit_line):
        w = LANE_WIDTH
        far_entry_p =   get_offset_point_at_distance(entry_line, entry_line.length - 70.0, LANE_WIDTH/2)
        entry_p2 =      get_offset_point_at_distance(entry_line, entry_line.length - w - 0.1, LANE_WIDTH/2) # Control point to ensure spline orientation with street
        entry_p =       get_offset_point_at_distance(entry_line, entry_line.length - w, LANE_WIDTH/2)
        exit_p =        get_offset_point_at_distance(exit_line, w, LANE_WIDTH/2)
        exit_p2 =       get_offset_point_at_distance(exit_line, w + 0.1, LANE_WIDTH/2)
        far_exit_p =    get_offset_point_at_distance(exit_line, 70.0, LANE_WIDTH/2)
        coords = [  list(far_entry_p.coords)[0],
                    list(entry_p2.coords)[0],
                    list(entry_p.coords)[0],
                    list(exit_p.coords)[0],
                    list(exit_p2.coords)[0],
                    list(far_exit_p.coords)[0]]
        x, y = zip(*coords)
        # Make a parametric quadratic spline with given knot vector
        x2, y2 = parametric_combined_spline(x, y, k=2, s=0.0)
        interpolating_spline_line = LineString(zip(x2, y2))
        return interpolating_spline_line

class AlhajyaseenAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Alhajyaseen Algorithm'

    def predict(self, sample):
        features = self._calculate_intersection_features(sample)
        A_1, A_2, R_min, V_min = self._calculate_curve_parameters(features, sample)
        print "A_1:", A_1
        print "A_2:", A_2
        print "R_min:", R_min
        print "V_min:", V_min

    def _calculate_intersection_features(self, sample):
        # All variables are calculated as if being in left hand traffic
        # Other variables
        intersection_angle = sample['X'][_feature_types.index('intersection_angle')] # intersection angle in radians (different system than Alhajyaseen)

        # Alhajyaseen features
        features = {}
        features['R_c'] = 3.0                       # Corner radius [m]
        features['theta'] = np.degrees(np.pi - np.abs(intersection_angle)) # intersection angle [deg]
        features['heavy_vehicle_dummy'] = 0.0       # Passenger car
        features['V_in'] = 25.0                     # Approaching speed estimated [km/h]

        if intersection_angle >= 0.0:
            # Right turn in left hand traffic
            features['D_HN_IN'] = INT_DIST + LANE_WIDTH/2.0     # Distance from IP to hard nose at entry [m]
            features['D_HN_OUT'] = INT_DIST + LANE_WIDTH/2.0    # Distance from IP to hard nose at exit [m]
            features['MIN_D_HN'] = min(features['D_HN_IN'], features['D_HN_OUT'])    # Minimum of the two distances
        else:
            # Left turn in left hand traffic
            features['lateral_exit_shoulder_dist'] = LANE_WIDTH/2.0
        return features

    def _calculate_curve_parameters(self, f, sample):
        intersection_angle = sample['X'][_feature_types.index('intersection_angle')] # intersection angle in radians (different system than Alhajyaseen)
        if intersection_angle >= 0.0:
            # Right turn in left hand traffic
            V_min = 4.49 \
                    + 0.072*f['theta'] \
                    + 0.0092*f['D_HN_IN'] \
                    + 0.105*f['D_HN_OUT'] \
                    + 0.38*f['V_in']

            R_min = 1.86 \
                    + 0.062*f['theta'] \
                    + 0.13*f['MIN_D_HN'] \
                    + 0.36*V_min

            A_1   = - 8.65 \
                    + 0.17*f['D_HN_IN'] \
                    + 0.29*V_min

            A_2   = 3.63 \
                    + 0.24*f['D_HN_OUT'] \
                    + 0.29*V_min
        else:
            # Left turn in left hand traffic
            V_min = - 1.08 \
                    + 0.22*f['R_c'] \
                    + 0.14*f['theta'] \
                    - 1.79*f['heavy_vehicle_dummy'] \
                    + 0.84*f['lateral_exit_shoulder_dist'] \
                    + 0.091*f['V_in']

            R_min = - 6.46 \
                    + 0.39*f['R_c'] \
                    + 0.13*f['theta'] \
                    + 0.86*f['lateral_exit_shoulder_dist']

            A_1   = - 1.65 \
                    + 0.33*f['R_c'] \
                    + 0.0404*f['theta'] \
                    + 0.46*f['lateral_exit_shoulder_dist'] \
                    + 0.37*V_min

            A_2   = 2.33 \
                    + 0.34*f['R_c'] \
                    + 2.051*f['heavy_vehicle_dummy'] \
                    + 1.041*f['lateral_exit_shoulder_dist'] \
                    + 0.27*V_min

        return A_1, A_2, R_min, V_min
