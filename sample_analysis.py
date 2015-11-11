#!/usr/bin/python
#coding:utf-8
from __future__ import division
import automatic_test
import numpy as np
import extract_features
from extract_features import _feature_types
# import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
import pandas
from constants import INT_DIST
import plot_helper
import sys

# matplotlib.rcParams.update({'font.size': 28})
sns.set_context("paper", font_scale=1.8)
sns.plt.rc("figure", figsize=[12,6])

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

def find_closest_point_along_vec(line, p, vec):
    SEARCH_LENGTH = 1000.0
    pos_ruler = LineString([p.coords[0], list(np.array(p.coords[0]) + vec*SEARCH_LENGTH)])
    neg_ruler = LineString([p.coords[0], list(np.array(p.coords[0]) - vec*SEARCH_LENGTH)])
    pos_p = extract_features.find_closest_intersection(pos_ruler, p, line)
    neg_p = extract_features.find_closest_intersection(neg_ruler, p, line)

    if pos_p != None:
        dist_n = p.distance(pos_p)
    else:
        dist_n = None
    if neg_p != None:
        dist_nn = p.distance(neg_p)
    else:
        dist_nn = None

    if dist_n != None and dist_nn != None:
        if dist_n <= dist_nn:
            return pos_p
        else:
            return neg_p
    if dist_n == dist_nn == None:
        raise extract_features.NoIntersectionError("No intersection of normals with track found")
    else:
        if dist_n != None: return pos_p
        else: return neg_p

def split_path_at_line_dist(path, way_line, vec, dist):
    """Split a path at the projected point along vec at dist of way_line"""
    way_p = way_line.interpolate(dist)

    path_p = find_closest_point_along_vec(path, way_p, vec)
    path_split_dist = path.project(path_p)
    return extract_features.split_line(path, path_split_dist)

def plot_sample_intersection_curvature(samples, title="Sample curvature over intersection coordinates", ax=None, color=None):
    """Plot each sample's curvature relative to the intersection distances coordinate system"""
    print "Curvature calculation..."
    sample_steps = 100
    curvatures = np.zeros((len(samples), sample_steps))
    line_dists = np.array(curvatures)

    for i, s in enumerate(samples):
        track_line = s['geometry']['track_line']
        entry_line = s['geometry']['entry_line']
        exit_line = s['geometry']['exit_line']
        try:
            half_angle_vec = extract_features.get_half_angle_vec(exit_line, s['X'][_feature_types.index('intersection_angle')])
            # Limit path to a set s_di interval at intersection
            # _, track_line = split_path_at_line_dist(track_line, entry_line, half_angle_vec, entry_line.length-36.0)
            # track_line, _ = split_path_at_line_dist(track_line, exit_line, half_angle_vec, 36.0)

            curvature_sample_coords = [track_line.interpolate(dist).coords[0] for dist in np.linspace(0, track_line.length, sample_steps)]
            X, Y = zip(*curvature_sample_coords)


            way_line, dists = extract_features.set_up_way_line_and_distances(entry_line, exit_line)
            way_line = extract_features.extend_line(way_line, 1000.0, direction="both") # Make sure the way_line is not too short to cover the whole track
            LineDistances, _ = extract_features.get_distances_from_cartesian(X, Y, way_line, half_angle_vec)
            line_dists[i] = LineDistances - 1000.0 - INT_DIST  # Shift to the actual coordinate system
            curvatures[i] = extract_features.get_line_curvature(track_line, sample_steps)
        except extract_features.NoIntersectionError as e:
            #plot_helper.plot_intersection(s, additional_lines=[way_line])
            print e
            continue


    # fig = plt.figure()
    # sns.plt.hold(True)
    for i in range(curvatures.shape[0]):
        ax.plot(line_dists[i], curvatures[i], color=color, linestyle='-')
    # plt.title(title)
    # sns.plt.show()

def show_bar_plot():
    pass

if __name__ == "__main__":

    dataset_samples = [("KITTI + Karlsruhe", automatic_test.load_samples("data/training_data/samples_analysis/samples_kitti.pickle")),
                        ("Darmstadt", automatic_test.load_samples("data/training_data/samples_analysis/samples_darmstadt.pickle"))]


    # Intersection angles
    figure1, axes1 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes1[i]
        # sns.set_style("whitegrid")
        intersection_angles = get_array_from_feature(samples, 'intersection_angle')/(np.pi)*180.0
        sns.distplot(intersection_angles, bins=20, kde=False, rug=True, ax=ax)
        # sns.plt.bar(intersection_angles, bins=20)
        ax.set_xlabel("Kreuzungswinkel")
        ax.set_ylabel("Anzahl der Kreuzungen")
    # sns.plt.show(figure1)

    # Oneways
    figure2, axes2 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes2[i]
        oneways_entry = [extract_features.float_to_boolean(f) for f in get_array_from_feature(samples, 'oneway_entry')]
        oneways_exit = [extract_features.float_to_boolean(f) for f in get_array_from_feature(samples, 'oneway_exit')]
        oneways = [en and ex for en, ex in zip(oneways_entry, oneways_exit)]

        print "Kreuzungen mit Einbahnstraßen: %d/%d" % (oneways.count(True), len(samples))
        oneway = {True:u"Einbahnstraße", False:u"Gegenverkehrsstraße"}
        # sns.set_style("whitegrid")
        sns.countplot([oneway[o] for o in oneways], order=[u"Einbahnstraße", u"Gegenverkehrsstraße"], ax=ax)
        if i == 0:
            ax.set_ylabel("Kreuzungen mit entsprechenden Armen")
        else:
            ax.set_ylabel("")
    # sns.plt.show(figure2)

    def label_int(ms):
        if ms != 0.0:
            return str(int(ms))
        else:
            return "n/a"
    def get_order(l):
        return [label_int(ms) for ms in sorted(list(set(l)))]

    # Maxspeeds
    figure3, axes3 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes3[i]
        maxspeeds = list(get_array_from_feature(samples, 'maxspeed_entry')) + list(get_array_from_feature(samples, 'maxspeed_exit'))
        maxspeeds_labels = [label_int(ms) for ms in maxspeeds]
        maxspeed_plot = sns.countplot(maxspeeds_labels, order=["n/a","30","50"], ax=ax)
        ax.set_xlabel(u"Erlaubte Höchstgeschwindigkeit [km/h]")
        if i == 0:
            ax.set_ylabel("Anzahl der Kreuzungsarme")
        else:
            ax.set_ylabel("")
    # sns.plt.show(figure3)

    # Lanes
    figure4, axes4 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes4[i]
        lanes = list(get_array_from_feature(samples, 'lane_count_entry')) + list(get_array_from_feature(samples, 'lane_count_exit'))
        lanes_labels = [label_int(ms) for ms in lanes]
        lanes_plot = sns.countplot(lanes_labels, order=["n/a", "1", "2", "4"], ax=ax)
        ax.set_xlabel(u"Fahrstreifenanzahl")
        if i == 0:
            ax.set_ylabel("Anzahl der Kreuzungsarme")
        else:
            ax.set_ylabel("")
        # sns.plt.title("KITTI + Karlsruhe")
    sns.plt.show(figure4)
    # # Track curvatures
    # figure5, axes5 = sns.plt.subplots(1, 2)
    # for i, (name, samples) in enumerate(dataset_samples):
    #     l_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] >= 0.]
    #     r_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] < 0.]
    #     ax = axes5[i]
    #     ax.hold(True)
    #     plot_sample_intersection_curvature(l_samples, ax=ax, color=(0,0,1,0.5))
    #     plot_sample_intersection_curvature(r_samples, ax=ax, color=(1,0,0,0.5))
    #     ax.set_xlim([-35,35])
    #     ax.set_ylim([-0.2,0.2])
    # sns.plt.show(figure5)

    figure6, axes6 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes6[i]
        maxspeed_entry = get_array_from_feature(samples, 'maxspeed_entry')
        maxspeed_exit = get_array_from_feature(samples, 'maxspeed_exit')
        vehicle_speed_entry = get_array_from_feature(samples, 'vehicle_speed_entry')
        vehicle_speed_exit = get_array_from_feature(samples, 'vehicle_speed_exit')
        # mean_err_entry = np.mean(np.abs(vehicle_speed_entry-maxspeed_entry))
        # std_err_entry = np.std(np.abs(vehicle_speed_entry-maxspeed_entry))
        # mean_err_exit = np.mean(np.abs(vehicle_speed_exit-maxspeed_exit))
        # std_err_exit = np.std(np.abs(vehicle_speed_exit-maxspeed_exit))
        sns.distplot(vehicle_speed_entry - maxspeed_entry, bins=20, kde=False, rug=True, ax=ax)
        ax.set_xlabel("Geschwindigkeitsfehler")
    sns.plt.show(figure6)
