#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from extract_features import extended_interpolate, get_normal_to_line, \
            sample_line, _feature_types, get_offset_point_at_distance, extend_line, \
            sample_line_all
from constants import LANE_WIDTH
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
