#!/usr/bin/python
#coding:utf-8
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import pandas
import numpy as np
from extract_features import get_normal_to_line, extend_line, get_predicted_line,\
                _feature_types, get_cartesian_from_polar, get_angle_between_lines,\
                rotate_xy, set_up_way_line_and_distances, get_cartesian_from_distances
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely import affinity
import copy
from constants import SAMPLE_RESOLUTION, INT_DIST
import scipy.interpolate
import scipy.stats

def get_distributed_colors(number, colormap='Set1'):
    cmap = plt.get_cmap('Set1')
    rgbcolors = [cmap(i) for i in np.linspace(0., 1., number)]
    return rgbcolors

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

class RBFValley:
    """Constructs a 'valley' of RBFs at node points and can be evaluated at
    any given point"""
    def __init__(self, X, Y, epsilon):
        self._nX = X
        self._nY = Y
        self._epsilon = epsilon

    def __call__(self, x, y):
        R = np.sqrt(np.power(self._nX - x, 2) + np.power(self._nY - y, 2))
        return np.sum(np.exp(- np.power(self._epsilon * R, 2)))

def get_heatmap_from_polar_all_predictors(predictions, curve_secant, intersection_angle):
    # Set up the RBF Interpolator
    angle_steps = np.linspace(0., np.pi, SAMPLE_RESOLUTION)
    rbfPhi = []
    rbfR = []
    for pred in predictions:
        rbfPhi.extend(list(angle_steps))
        rbfR.extend(list(pred))
    rbfi = RBFValley(rbfPhi, rbfR, 1.0)

    # Set up the grid to sample the RBF at
    r_min = 2.0
    r_max = 30.0
    r_resolution = 60
    phi_min = 0.0
    phi_max = np.pi
    phi_resolution = 100
    R = np.rot90(np.tile(np.linspace(r_min, r_max, r_resolution), (phi_resolution, 1)))
    Phi = np.tile(np.linspace(phi_min, phi_max, phi_resolution), (r_resolution, 1))
    D = np.zeros((np.shape(Phi)[0]-1, np.shape(Phi)[1]-1))

    # Sample the RBF
    for j in range(np.shape(Phi)[0]-1):
        for k in range(np.shape(Phi)[1]-1):
            D[j,k] = rbfi(Phi[j,k], R[j,k])

    # Transform RBF grid into XY-Space for heatmap
    X, Y = get_cartesian_from_polar(R, Phi, curve_secant, intersection_angle)

    return X, Y, D

def get_heatmap_from_distances_all_predictors(predictions, entry_line, exit_line, half_angle_vec):
    # Set up the RBF Interpolator
    way_line, line_distances = set_up_way_line_and_distances(entry_line, exit_line)
    rbfLineDist = []
    rbfMeasureDist = []
    for pred in predictions:
        rbfLineDist.extend(line_distances)
        rbfMeasureDist.extend(pred)
    rbfi = RBFValley(rbfLineDist, rbfMeasureDist, 0.5)

    # Set up the grid to sample the RBF at
    line_dist_min = 0.
    line_dist_max = 2*INT_DIST
    line_dist_resolution = 100
    measure_dist_min = -10.
    measure_dist_max = 10.
    measure_dist_resolution = 60
    LineDists = np.rot90(np.tile(np.linspace(line_dist_min, line_dist_max, line_dist_resolution), (measure_dist_resolution, 1)))
    MeasureDists = np.tile(np.linspace(measure_dist_min, measure_dist_max, measure_dist_resolution), (line_dist_resolution, 1))
    D = np.zeros((np.shape(LineDists)[0]-1, np.shape(MeasureDists)[1]-1))

    # Sample the RBF
    for j in range(np.shape(LineDists)[0]-1):
        for k in range(np.shape(LineDists)[1]-1):
            D[j,k] = rbfi(LineDists[j,k], MeasureDists[j,k])

    # Transform RBF grid into XY-Space for heatmap
    X, Y = get_cartesian_from_distances(LineDists, MeasureDists, way_line, half_angle_vec)

    return X, Y, D

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

def plot_intersection(sample, predicted=[], rgbcolors=[], labels=[], label_methods=[], heatmap=None, title=None, block=True, orientation="preserve"):
    # normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False, direction="both")
    # normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False, direction="both")
    csample = copy.deepcopy(sample)
    entry_line =            csample['geometry']['entry_line']
    exit_line =             csample['geometry']['exit_line']
    curve_secant =          csample['geometry']['curve_secant']
    track_line =            csample['geometry']['track_line']
    half_angle_line =       csample['geometry']['half_angle_line']
    intersection_angle =    csample['X'][_feature_types.index('intersection_angle')]

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
        # And update in copied sample to be used by submethods
        entry_line = affinity.rotate(entry_line, phi, origin=rot_c, use_radians=True)
        csample['geometry']['entry_line'] = entry_line
        exit_line = affinity.rotate(exit_line, phi, origin=rot_c, use_radians=True)
        csample['geometry']['exit_line'] = exit_line
        curve_secant = affinity.rotate(curve_secant, phi, origin=rot_c, use_radians=True)
        csample['geometry']['curve_secant'] = curve_secant
        track_line = affinity.rotate(track_line, phi, origin=rot_c, use_radians=True)
        csample['geometry']['track_line'] = track_line
        half_angle_line = affinity.rotate(half_angle_line, phi, origin=rot_c, use_radians=True)
        csample['geometry']['half_angle_line'] = half_angle_line

    handles = []
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')

    if heatmap != None:
        X, Y, D = heatmap
        # Rotate if necessary
        if rotation[0] != 0.:
            rot_phi, rot_c = rotation
            target_shape = np.shape(X)
            coords = np.transpose(np.vstack((X.ravel(), Y.ravel())))
            rot_coords = np.transpose(rotate_xy(coords, rot_phi, rot_c))
            X = np.reshape(rot_coords[0], target_shape)
            Y = np.reshape(rot_coords[1], target_shape)

        ax = plt.gca()
        p = ax.pcolormesh(X, Y, D, cmap="Oranges")
        plt.gcf().colorbar(p)

    plot_lines('k', entry_line, exit_line, half_angle_line)
    # plot_line('m', normal_en, normal_ex)
    # plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('k', curve_secant)
    handles.append( plot_line('r', track_line, 'Measured Track') )
    # handles.append( plot_line('b', get_predicted_line(csample, csample['y']), 'Measured Track Distances') )
    plot_arrows_along_line('r', track_line)
    # plot_arrows_along_line('b', get_predicted_line(csample, csample['y']))

    if predicted:
        if rgbcolors == []:
            rgbcolors = get_distributed_colors(len(predicted))
        if labels == []:
            labels = [""]*len(predicted)
        if label_methods == []:
            label_methods = [sample['label']['selected_method']]*len(predicted)
        for pred, color, label, label_method in zip(predicted, rgbcolors, labels, label_methods):
            # print sample['y'] - pred
            line = get_predicted_line(pred, label_method, csample)
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
