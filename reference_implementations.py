#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from extract_features import extended_interpolate, get_normal_to_line
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

def get_geiger_line(entry_line, exit_line):
    # w = 2 * LANE_WIDTH
    w = 30.0
    center_p = exit_line.interpolate(0.)
    far_entry_n =   get_normal_to_line(entry_line, entry_line.length - 70.0)
    entry_n =       get_normal_to_line(entry_line, entry_line.length - w)
    far_exit_n =    get_normal_to_line(exit_line, 70.0)
    exit_n =        get_normal_to_line(exit_line, w)
    far_entry_p =   extended_interpolate(far_entry_n, LANE_WIDTH/2)
    entry_p =       extended_interpolate(entry_n, LANE_WIDTH/2)
    far_exit_p =    extended_interpolate(far_exit_n, LANE_WIDTH/2)
    exit_p =        extended_interpolate(exit_n, LANE_WIDTH/2)
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

class GeigerAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Quadratic B-Spline (Geiger 2014)'

    def predict(self, test_sample):
        return get_geiger_line(test_sample['geometry']['entry_line'], test_sample['geometry']['exit_line'])

def get_interpolating_spline_line(entry_line, exit_line):
    w = LANE_WIDTH
    far_entry_n =   get_normal_to_line(entry_line, entry_line.length - 70.0)
    entry_n2 =      get_normal_to_line(entry_line, entry_line.length - w - 0.1)
    entry_n =       get_normal_to_line(entry_line, entry_line.length - w)
    exit_n =        get_normal_to_line(exit_line, w)
    exit_n2 =       get_normal_to_line(exit_line, w + 0.1)
    far_exit_n =    get_normal_to_line(exit_line, 70.0)
    far_entry_p =   extended_interpolate(far_entry_n, LANE_WIDTH/2)
    entry_p2 =      extended_interpolate(entry_n2, LANE_WIDTH/2) # Control point to ensure spline orientation with street
    entry_p =       extended_interpolate(entry_n, LANE_WIDTH/2)
    exit_p =        extended_interpolate(exit_n, LANE_WIDTH/2)
    exit_p2 =       extended_interpolate(exit_n2, LANE_WIDTH/2)
    far_exit_p =    extended_interpolate(far_exit_n, LANE_WIDTH/2)
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

class InterpolatingSplineAlgorithm(automatic_test.PredictionAlgorithm):
    def __init__(self):
        self.name = 'Standard Interpolating Spline (k=2)'

    def predict(self, test_sample):
        return get_interpolating_spline_line(test_sample['geometry']['entry_line'], test_sample['geometry']['exit_line'])
