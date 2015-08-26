#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from extract_features import extended_interpolate
from constants import LANE_WIDTH

def parametric_spline(x, y, k=3, resolution=100):
    """Return a parametric univariate spline through x, y evaluated at resolution"""
    x = np.array(x)
    y = np.array(y)

    nt = np.linspace(0, 1, resolution)
    t = np.zeros(x.shape)
    # Calculate the partial distances between coordinates
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    # Sum up the partial distances to have the absolute distance from start point
    t = np.cumsum(t)
    t /= t[-1]
    x_spl = scipy.interpolate.UnivariateSpline(t, x, k=k)
    y_spl = scipy.interpolate.UnivariateSpline(t, y, k=k)
    x2 = x_spl(nt)
    y2 = y_spl(nt)
    return x2, y2

def geiger_path(entry_line, exit_line, start_dist):
    center_p = exit_line.interpolate(0.)
    entry_p = extended_interpolate(entry_line, entry_line.length - start_dist)
    exit_p = extended_interpolate(exit_line, start_dist)
    coords = [list(entry_p.coords)[0], list(center_p.coords)[0], list(exit_p.coords)[0]]
    x, y = zip(*coords)
    x2, y2 = parametric_spline(x, y, k=2)
    geiger_path_line = LineString(zip(x2, y2))
    return geiger_path_line

def kuhnt_path():
    pass
