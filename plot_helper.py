#!/usr/bin/python
#coding:utf-8
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas
import numpy as np
from extract_features import get_normal_to_line, extend_line, get_predicted_line,\
                _feature_types, get_cartesian_from_polar, get_angle_between_lines,\
                rotate_xy
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely import affinity

def get_distributed_colors(number, colormap='Set1'):
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0., 1., number)]
    return colors

def plot_coords(x, y, color, label=None):
    handle, = plt.plot(x,y, color=color, linestyle='-', label=label)
    return handle

def plot_line(color, line, label=None):
    coords = list(line.coords)
    x,y = zip(*coords)
    handle = plot_coords(x, y, color, label=label)
    return handle

def plot_lines(color, *lines):
    handles = []
    for l in lines:
        handles.append(plot_line(color, l))
    return handles

def plot_arrow(color, center_line, dist, normalized=False):
    ARROW_LENGTH = 5.0
    origin_p = center_line.interpolate(dist, normalized=normalized)
    normal_line = get_normal_to_line(center_line, dist, normalized=normalized)
    half_arrow = extend_line(normal_line, ARROW_LENGTH - normal_line.length, direction="forward")
    half_arrow = affinity.rotate(half_arrow, -30.0, origin=origin_p)
    plot_line(color, half_arrow)
    half_arrow = affinity.rotate(half_arrow, -105.0, origin=origin_p)
    plot_line(color, half_arrow)

def plot_arrows_along_line(color, center_line):
    MIN_DIST = 50.0
    arrow_count = int(center_line.length / MIN_DIST)
    for i in range(1, arrow_count + 1):
        plot_arrow(color, center_line, i*MIN_DIST, normalized=False)

def plot_polar_probability_heatmap(predicted_proba, curve_secant, intersection_angle):
    """Plot a heatmap in polar coordinates"""
    prediction =    np.flipud(np.rot90(predicted_proba['predictions_proba']))
    bin_num =       np.shape(prediction)[0]
    max_radius =    predicted_proba['max_radius']
    min_radius =    predicted_proba['min_radius']
    bin_width =     np.pi/(np.shape(prediction)[1])
    R =             np.rot90(np.tile(np.linspace(max_radius, min_radius, bin_num + 1), (np.shape(prediction)[1] + 1, 1)))
    Phi =           np.tile((np.linspace(0.-bin_width/2, np.pi+bin_width/2, np.shape(prediction)[1] + 1) - bin_width/2), (bin_num + 1, 1))
    X, Y =          get_cartesian_from_polar(R, Phi, curve_secant, intersection_angle)
    # if rotation:
    #     rot_phi, rot_c = rotation
    #     target_shape = np.shape(X)
    #     coords = np.transpose(np.vstack((X.ravel(), Y.ravel())))
    #     rot_coords = np.transpose(rotate_xy(coords, rot_phi, rot_c))
    #     X = np.reshape(rot_coords[0], target_shape)
    #     Y = np.reshape(rot_coords[1], target_shape)
    ax = plt.gca()
    p = ax.pcolormesh(X, Y, prediction, cmap="Oranges")
    plt.gcf().colorbar(p)

def plot_intersection(sample, predicted_radii=[], predicted_proba=[], labels=[], title=None, block=True, probability_map=None, orientation="preserve"):
    # normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False, direction="both")
    # normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False, direction="both")
    entry_line =            sample['geometry']['entry_line']
    exit_line =             sample['geometry']['exit_line']
    curve_secant =          sample['geometry']['curve_secant']
    track_line =            sample['geometry']['track_line']
    intersection_angle =    sample['X'][_feature_types.index('intersection_angle')]

    rotation = (0., (0.,0.)) # rad
    if orientation == "curve-secant":
        x_axis = LineString([(0,0),(1,0)])
        inv_curve_secant = LineString([curve_secant.coords[1], curve_secant.coords[0]])
        phi = - get_angle_between_lines(x_axis, curve_secant)
        if intersection_angle > 0.:
            # With the intersection angle it can be determined how the
            # intersection is upright (curve_secant is below entry and exit_line)
            phi = np.pi + phi
        rot_c, = curve_secant.interpolate(0.5, normalized=True).coords[:]
        rotation = (phi, rot_c)
    phi, rot_c = rotation
    if rotation[0] != 0.:
        # Rotate all given LineStrings
        entry_line = affinity.rotate(entry_line, phi, origin=rot_c, use_radians=True)
        exit_line = affinity.rotate(exit_line, phi, origin=rot_c, use_radians=True)
        curve_secant = affinity.rotate(curve_secant, phi, origin=rot_c, use_radians=True)
        track_line = affinity.rotate(track_line, phi, origin=rot_c, use_radians=True)

    handles = []
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')

    for proba_map in predicted_proba:
        plot_polar_probability_heatmap(proba_map, curve_secant, intersection_angle)

    plot_lines('k', entry_line, exit_line)
    # plot_line('m', normal_en, normal_ex)
    # plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('k', curve_secant)
    handles.append( plot_line('r', track_line, 'Measured Track') )
    plot_arrows_along_line('r', track_line)

    colors = get_distributed_colors(len(predicted_radii))
    for radii, color, label in zip(predicted_radii, colors, labels):
        line = get_predicted_line(curve_secant, radii, intersection_angle)
        handles.append( plot_line(color, line, label) )
    plt.legend(handles=handles)
    if title: plt.title(title)
    plt.show(block=block)

def plot_probability_heatmap(predicted_proba):
    """Plot a heatmap in cartesian coordinates"""
    prediction =    np.rot90(predicted_proba['predictions_proba'])
    bin_num =       np.shape(prediction)[0]
    max_radius =    predicted_proba['max_radius']
    min_radius =    predicted_proba['min_radius']
    bin_width =     180./(np.shape(prediction)[1])
    angle_steps =   np.linspace(0., 180.+bin_width, np.shape(prediction)[1] + 1) - bin_width/2
    radius_steps =  np.linspace(max_radius, min_radius, bin_num + 1)
    ax = plt.gca()
    p = ax.pcolormesh(angle_steps, radius_steps, prediction, cmap="Oranges")
    plt.gcf().colorbar(p)

def plot_graph(track_radii, predicted_radii, predicted_proba=[], labels=[], title=None):
    angle_steps = np.linspace(0., 180., len(predicted_radii[0]))
    handles = []
    fig = plt.figure()
    plt.hold(True)
    ax = plt.gca()
    for proba_map in predicted_proba:
        plot_probability_heatmap(proba_map)
    colors = get_distributed_colors(len(predicted_radii))
    handles.append( plot_coords(angle_steps, track_radii, "red", "Measured Track") )
    for radii, color, label in zip(predicted_radii, colors, labels):
        handles.append( plot_coords(angle_steps, radii, color, label) )
    plt.legend(handles=handles)
    if title: plt.title(title)
    plt.show()

def plot_sampled_track(label):
    fig = plt.figure()
    plt.hold(True)
    plt.plot(label["angles"],label["radii"],'b.-')
    plt.show()
