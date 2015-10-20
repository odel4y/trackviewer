#!/usr/bin/python
#coding:utf-8
import automatic_test
import numpy as np
import extract_features
from extract_features import _feature_types
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from constants import INT_DIST
import plot_helper

def get_array_from_feature(samples, feature):
    feature_i = extract_features._feature_types.index(feature)
    return np.array([s['X'][feature_i] for s in samples])

def plot_label_heatmap(samples, bars_y = 30):
    angles = np.linspace(0., np.pi, len(samples[0]['y']))
    min_x = np.amin(angles)
    max_x = np.amax(angles)
    radii = np.zeros((len(samples), len(angles)))
    for i, s in enumerate(samples):
        radii[i] = s['y']
    min_y = np.amin(radii)
    max_y = np.amax(radii)
    heatmap_array = np.zeros((bars_y, len(angles)))
    for i in xrange(np.shape(radii)[0]):
        for j in xrange(np.shape(radii)[1]):
            dest_j = j
            dest_i = round((radii[i,j] - min_y) / (max_y - min_y) * bars_y)-1
            heatmap_array[dest_i, dest_j] += 1
    indices = np.linspace(min_y, max_y, bars_y)
    indices = ["%.1f" % i for i in indices]
    columns = np.linspace(0., 180., len(angles))
    columns = ["%.1f" % i for i in columns]
    heatmap_array = np.flipud(heatmap_array)
    heatmap_frame = pandas.DataFrame(data=heatmap_array, index=reversed(indices), columns=columns)
    f = sns.heatmap(heatmap_frame)
    plt.subplots_adjust(top=0.9)
    plt.title("Labels Heatmap")
    sns.axlabel("Angle", "Radius")
    sns.plt.show(f)

def plot_sample_intersection_curvature(samples, title="Sample curvature over intersection coordinates"):
    """Plot each sample's curvature relative to the intersection distances coordinate system"""
    sample_steps = 100
    curvatures = np.zeros((len(samples), sample_steps))
    line_dists = np.array(curvatures)

    for i, s in enumerate(samples):
        track_line = s['geometry']['track_line']
        curvature_sample_coords = [track_line.interpolate(dist).coords[0] for dist in np.linspace(0, track_line.length, 100)]
        X, Y = zip(*curvature_sample_coords)

        half_angle_vec = extract_features.get_half_angle_vec(s['geometry']['exit_line'], s['X'][_feature_types.index('intersection_angle')])

        way_line, dists = extract_features.set_up_way_line_and_distances(s['geometry']['entry_line'], s['geometry']['exit_line'])
        way_line = extract_features.extend_line(way_line, 1000.0, direction="both") # Make sure the way_line is not too short to cover the whole track
        try:
            LineDistances, _ = extract_features.get_distances_from_cartesian(X, Y, way_line, half_angle_vec)
        except extract_features.NoIntersectionError:
            plot_helper.plot_intersection(s, additional_lines=[way_line])

        line_dists[i] = LineDistances - 1000.0 - INT_DIST  # Shift to the actual coordinate system
        curvatures[i] = extract_features.get_line_curvature(track_line, sample_steps)

    fig = plt.figure()
    plt.hold(True)
    for i in range(curvatures.shape[0]):
        plt.plot(line_dists[i], curvatures[i], color=(.5,.5,.5), linestyle='-')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    samples = automatic_test.load_samples('data/training_data/samples.pickle')
    print 'Sample count:', len(samples)
    oneway_entry = list(get_array_from_feature(samples, 'oneway_entry'))
    print 'oneway_entry: Yes: %d No: %d' % (oneway_entry.count(1.0), oneway_entry.count(-1.0))
    oneway_exit = list(get_array_from_feature(samples, 'oneway_exit'))
    print 'oneway_exit: Yes: %d No: %d' % (oneway_exit.count(1.0), oneway_exit.count(-1.0))
    intersection_angles = get_array_from_feature(samples, 'intersection_angle')/(np.pi)*180.0
    sns.set(color_codes=True)
    plot_label_heatmap(samples)
    angle_plot = sns.distplot(intersection_angles, bins=20, kde=False, rug=True)
    sns.plt.show(angle_plot)
    plot_sample_intersection_curvature(samples)
