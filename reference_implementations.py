#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
import shapely.affinity
from extract_features import *
from extract_features import _feature_types
from constants import LANE_WIDTH, INT_DIST
import rectify_prepared_data
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
        predicted_line = self.get_interpolating_spline_line(sample)
        pred = sample_line_all(predicted_line,
                            sample['label']['selected_method'],
                            sample)
        return pred

    def get_interpolating_spline_line(self, sample):
        entry_line = sample['geometry']['entry_line']
        exit_line = sample['geometry']['exit_line']
        oneway_entry = float_to_boolean(sample['X'][_feature_types.index('oneway_entry')])
        oneway_exit = float_to_boolean(sample['X'][_feature_types.index('oneway_exit')])
        if oneway_entry:
            distance_entry = 0.
        else:
            distance_entry = LANE_WIDTH/2
        if oneway_exit:
            distance_exit = 0.
        else:
            distance_exit = LANE_WIDTH/2

        w = 2*LANE_WIDTH
        far_entry_p =   get_offset_point_at_distance(entry_line, entry_line.length - 70.0, distance_entry)
        entry_p2 =      get_offset_point_at_distance(entry_line, entry_line.length - w - 0.1, distance_entry) # Control point to ensure spline orientation with street
        entry_p =       get_offset_point_at_distance(entry_line, entry_line.length - w, distance_entry)
        exit_p =        get_offset_point_at_distance(exit_line, w, distance_exit)
        exit_p2 =       get_offset_point_at_distance(exit_line, w + 0.1, distance_exit)
        far_exit_p =    get_offset_point_at_distance(exit_line, 70.0, distance_exit)
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
    def __init__(self, allow_rectification=False, allow_actual_speed=True):
        self.name = 'Alhajyaseen Algorithm'
        self.allow_rectification = allow_rectification # Move the line to accurately fit entry and exit distance
        self.allow_actual_speed = allow_actual_speed # Use the actual vehicle entry speed

    def predict(self, sample):
        features = self._calculate_intersection_features(sample)
        A_1, A_2, R_min, V_min = self._calculate_curve_parameters(features, sample)

        curved_line = self._get_curved_line(A_1, A_2, R_min, sample)
        pred = sample_line_all(curved_line,
                            sample['label']['selected_method'],
                            sample)
        return pred

    def _get_curved_line(self, A_1, A_2, R_min, sample):
        intersection_angle = sample['X'][_feature_types.index('intersection_angle')]
        ang_sign = np.sign(intersection_angle)

        entry_line = sample['geometry']['entry_line']
        entry_lane_line = extend_line(entry_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="forward")
        exit_line = sample['geometry']['exit_line']
        exit_lane_line = extend_line(exit_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="backward")

        # Entering straight
        entry_tangent_vec = get_tangent_vec_at(entry_line, entry_line.length)
        entry_straight_line = self._get_straight_line(entry_tangent_vec, 50.0)

        # Entering Euler spiral
        entry_euler_line = self._get_euler_spiral_line(R_min, ang_sign*A_1, entry_tangent_vec)

        # Exiting Euler spiral
        exit_tangent_vec = -get_tangent_vec_at(exit_line, 0.)
        exit_euler_line = self._get_euler_spiral_line(R_min, -ang_sign*A_2, exit_tangent_vec)

        # Constant curvature segment
        normal1 = ang_sign*get_normal_vec_at(entry_euler_line, entry_euler_line.length)
        normal2 = -ang_sign*get_normal_vec_at(exit_euler_line, exit_euler_line.length)
        circular_line = self._get_circular_line(R_min, normal1, normal2)

        # Exiting straight line
        exit_straight_line = self._get_straight_line(-exit_tangent_vec, 50.0)

        # Join segments
        curved_line = self._join_segments(entry_straight_line, entry_euler_line, circular_line, exit_euler_line, exit_straight_line)

        # Place in intersection
        curved_line = self._place_line_in_intersection(curved_line, sample)

        return curved_line

    def _place_line_in_intersection(self, curved_line, sample):
        # Move line to approximate location of intersection
        entry_line = sample['geometry']['entry_line']
        curve_peak_coord = np.array(curved_line.interpolate(0.5, normalized=True).coords[0])
        translation_vec = np.array(entry_line.coords[-1]) - curve_peak_coord
        curved_line = shapely.affinity.translate(curved_line, xoff=translation_vec[0], yoff=translation_vec[1])

        # Place exactly in intersection
        if self.allow_rectification:
            desired_entry_distance = sample['X'][_feature_types.index('lane_distance_entry_projected_normal')]
            desired_exit_distance = sample['X'][_feature_types.index('lane_distance_exit_projected_normal')]
        else:
            desired_entry_distance = None
            desired_exit_distance = None
        curved_line = rectify_prepared_data.rectify_line(curved_line, sample, desired_entry_distance, desired_exit_distance)
        return curved_line

    def _join_segments(self, entry_straight_line, entry_euler_line, circular_line, exit_euler_line, exit_straight_line):
        entry_sl_coords = np.array(entry_straight_line.coords[:])
        entry_el_coords = np.array(entry_euler_line.coords[:])
        circ_l_coords = np.array(circular_line.coords[:])
        circ_l_coords = circ_l_coords - circ_l_coords[0]
        exit_el_coords = np.array(exit_euler_line.coords[:])
        exit_el_coords = np.flipud(exit_el_coords)
        exit_el_coords = exit_el_coords - exit_el_coords[0]
        exit_sl_coords = np.array(exit_straight_line.coords[:])

        joined_coords = np.vstack((
            entry_sl_coords,
            entry_el_coords[1:-1] + entry_sl_coords[1],
            circ_l_coords + entry_sl_coords[1] + entry_el_coords[-1],
            exit_el_coords[1:] + circ_l_coords[-1] + entry_el_coords[-1] + entry_sl_coords[1],
            exit_sl_coords + circ_l_coords[-1] + entry_el_coords[-1] + entry_sl_coords[1] + exit_el_coords[-1]
        ))

        return LineString([tuple(row) for row in joined_coords])

    def _get_euler_spiral_line(self, R_s, A, entry_tangent, sample_steps=100):
        L_s = np.power(A,2)/R_s
        ds = L_s/(sample_steps-1)

        coords = np.zeros((sample_steps,2))
        current_angle = get_absolute_vec_angle(entry_tangent)
        for j in range(1, sample_steps):
            last_curvature = (j-1)*ds/np.power(A,2)
            current_angle += ds*last_curvature*np.sign(A)
            step_vec = rotate_vec(np.array([1, 0]), current_angle)
            coords[j] = coords[j-1] + step_vec*ds
        return LineString([tuple(row) for row in coords])

    def _get_circular_line(self, R_min, entry_normal, exit_normal, sample_steps=100):
        phi = get_vec_angle(entry_normal, exit_normal)
        initial_phi = get_absolute_vec_angle(entry_normal)
        d_phi = phi/(sample_steps-1)

        coords = np.zeros((sample_steps,2))
        for j in range(sample_steps):
            current_angle = initial_phi + j*d_phi
            x = np.cos(current_angle)*R_min
            y = np.sin(current_angle)*R_min
            coords[j] = (x, y)

        return LineString([tuple(row) for row in coords])

    def _get_straight_line(self, entry_tangent, length):
        coords = np.zeros((2,2))
        entry_tangent = entry_tangent/np.linalg.norm(entry_tangent)
        coords[1] = entry_tangent*length

        return LineString([tuple(row) for row in coords])

    def _calculate_intersection_features(self, sample):
        # All variables are calculated as if being in left hand traffic
        # Other variables
        intersection_angle = sample['X'][_feature_types.index('intersection_angle')] # intersection angle in radians (different system than Alhajyaseen)

        # Alhajyaseen features
        features = {}
        features['R_c'] = 3.0                       # Corner radius [m]
        features['theta'] = np.degrees(np.pi - np.abs(intersection_angle)) # intersection angle [deg]
        features['heavy_vehicle_dummy'] = 0.0       # Passenger car
        if self.allow_actual_speed:
            features['V_in'] = sample['X'][_feature_types.index("vehicle_speed_entry")] # Approaching speed estimated [km/h]
        else:
            features['V_in'] = 30.0     # Dummy speed

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
