#!/usr/bin/python
#coding:utf-8
from __future__ import division
import numpy as np
import scipy.interpolate
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from extract_features import extended_interpolate, get_normal_to_line
from constants import LANE_WIDTH

def parametric_combined_spline(x, y, k=3, resolution=100, kv=None):
    """Return a linear combination of parametric univariate splines through x, y evaluated at resolution"""
    x = np.array(x)
    y = np.array(y)

    nt = np.linspace(0, 1, resolution)

    # Prepare linear combination of splines with given knot vector
    tckp,u = scipy.interpolate.splprep([x,y],k=k,t=kv)
    x2, y2 = scipy.interpolate.splev(np.linspace(0,1,400), tckp)

    return x2, y2

def parametric_spline(x, y, k=3, resolution=100, kv=None):
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
    print 'x:', t
    print 'kv:', kv
    if kv:
        # If a knot vector is given use it
        x_spl = scipy.interpolate.LSQUnivariateSpline(t, x, t=kv, k=k)
        y_spl = scipy.interpolate.LSQUnivariateSpline(t, y, t=kv, k=k)
    else:
        x_spl = scipy.interpolate.UnivariateSpline(t, x, k=k)
        y_spl = scipy.interpolate.UnivariateSpline(t, y, k=k)
    x2 = x_spl(nt)
    y2 = y_spl(nt)
    return x2, y2

def geiger_path(entry_line, exit_line):
    # w = 2 * LANE_WIDTH
    w = 10.0
    center_p = exit_line.interpolate(0.)
    far_entry_n =   get_normal_to_line(entry_line, entry_line.length - 70.0, direction="backward")
    entry_n =       get_normal_to_line(entry_line, entry_line.length - w, direction="backward")
    far_exit_n =    get_normal_to_line(exit_line, 70.0, direction="backward")
    exit_n =        get_normal_to_line(exit_line, w, direction="backward")
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

def kuhnt_path():
    pass
