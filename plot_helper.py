#!/usr/bin/python
#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from extract_features import get_normal_to_line, extend_line
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection

def get_distributed_colors(number, colormap='Set1'):
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0., 1., number)]
    return colors

def plot_line(color, line, label=None):
    coords = list(line.coords)
    x,y = zip(*coords)
    handle = plt.plot(x,y, color=c, linestyle='-', label=label)
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

def plot_intersection(entry_line, exit_line, curve_secant, track_line, predicted_lines=[], labels=[], title=None, block=True):
    # normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False, direction="both")
    # normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False, direction="both")
    handles = []
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')
    plot_line('k', entry_line, exit_line)
    # plot_line('m', normal_en, normal_ex)
    # plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('k', curve_secant)
    handles.append( plot_line('r', track_line, 'Measured Track') )
    plot_arrows_along_line('r', track_line)
    colors = get_distributed_colors(len(predicted_lines))
    for line, color, label in zip(predicted_lines, colors, labels):
        handles.append( plot_line(color, line, label) )
    plt.legend(handles=handles)
    plt.show(block=block)

def plot_sampled_track(label):
    fig = plt.figure()
    plt.hold(True)
    plt.plot(label["angles"],label["radii"],'b.-')
    plt.show()
