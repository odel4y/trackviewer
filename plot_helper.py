#!/usr/bin/python
#coding:utf-8
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas
import numpy as np
from extract_features import get_normal_to_line, extend_line
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

def plot_intersection(entry_line, exit_line, curve_secant, track_line, predicted_lines=[], labels=[], title=None, block=True, probability_map=None):
    # normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False, direction="both")
    # normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False, direction="both")
    handles = []
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')
    plot_lines('k', entry_line, exit_line)
    # plot_line('m', normal_en, normal_ex)
    # plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('k', curve_secant)
    handles.append( plot_line('r', track_line, 'Measured Track') )
    plot_arrows_along_line('r', track_line)
    colors = get_distributed_colors(len(predicted_lines))
    for line, color, label in zip(predicted_lines, colors, labels):
        handles.append( plot_line(color, line, label) )
    plt.legend(handles=handles)
    if title: plt.title(title)
    plt.show(block=block)

def plot_probability_heatmap(predicted_proba):
    prediction =    np.rot90(predicted_proba['predictions_proba'])
    print np.shape(prediction)
    bin_num =       np.shape(prediction)[0]
    max_radius =    predicted_proba['max_radius']
    min_radius =    predicted_proba['min_radius']
    angle_steps =   np.linspace(0., 180., np.shape(prediction)[1])
    radius_steps =  np.linspace(max_radius, min_radius, bin_num)
    # heatmap_frame = pandas.DataFrame(data=prediction, index=radius_steps, columns=angle_steps)
    heatmap_frame = pandas.DataFrame(data=prediction)
    ax = sns.heatmap(heatmap_frame, xticklabels=False, yticklabels=False)
    return ax

def plot_graph(track_radii, predicted_radii, predicted_proba, labels=[]):
    angle_steps = np.linspace(0., 180., len(track_radii))
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
    plt.show()

def plot_sampled_track(label):
    fig = plt.figure()
    plt.hold(True)
    plt.plot(label["angles"],label["radii"],'b.-')
    plt.show()
