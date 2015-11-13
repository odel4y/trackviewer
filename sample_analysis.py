#!/usr/bin/python
#coding:utf-8
from __future__ import division
import automatic_test
import numpy as np
import extract_features
from extract_features import *
from extract_features import _feature_types
# import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
import pandas
from constants import INT_DIST
import plot_helper
import sys
import pickle

# matplotlib.rcParams.update({'font.size': 28})
sns.set_context("paper", font_scale=1.8)
sns.plt.rc("figure", figsize=[12,6])

def get_average_speed_and_dist_of_sample(sample):
    fn = sample['pickled_filename']
    with open(fn, 'r') as f:
        int_sit = pickle.load(f)
    track = extract_features.transform_track_to_cartesian(int_sit['track'])
    track_line = sample['geometry']['track_line']
    entry_line = sample['geometry']['entry_line']
    exit_line = sample['geometry']['exit_line']

    dist_p1 = extended_interpolate(entry_line, entry_line.length-INT_DIST)
    normal1 = extend_line(get_normal_to_line(entry_line, entry_line.length-INT_DIST), 1000.0, direction="both")
    track_p1 = find_closest_intersection(normal1, dist_p1, track_line)
    track_i1 = find_nearest_coord_index(track_line, track_p1)

    dist_p2 = extended_interpolate(exit_line, INT_DIST)
    normal2 = extend_line(get_normal_to_line(exit_line, INT_DIST), 1000.0, direction="both")
    track_p2 = find_closest_intersection(normal2, dist_p2, track_line)
    track_i2 = find_nearest_coord_index(track_line, track_p2)

    time_delta = (track[track_i2][2] - track[track_i1][2]).total_seconds()
    min_i, max_i = min(track_i1, track_i2), max(track_i1, track_i2)
    dist = np.sum(np.linalg.norm(np.diff(np.array([(x, y) for (x, y, _) in track[min_i: max_i+1]]), axis=0), axis=1))

    return abs(dist/time_delta*3.6), dist

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
        handle, = ax.plot(line_dists[i], np.degrees(curvatures[i]), color=color, linestyle='-')
    return handle # Only need one
    # plt.title(title)
    # sns.plt.show()

def show_bar_plot():
    pass

if __name__ == "__main__":

    dataset_samples = [("KITTI + Karlsruhe", automatic_test.load_samples("data/training_data/samples_analysis/samples_kitti.pickle")),
                        ("Darmstadt", automatic_test.load_samples("data/training_data/samples_analysis/samples_darmstadt.pickle"))]

    sns.set_style("whitegrid", {"legend.frameon":True})
    # Intersection angles
    figure1, axes1 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes1[i]
        # sns.set_style("whitegrid")
        intersection_angles = list(get_array_from_feature(samples, 'intersection_angle')/(np.pi)*180.0)
        left_turn_count = len([ia for ia in intersection_angles if ia >= 0.])
        right_turn_count = len([ia for ia in intersection_angles if ia < 0.])
        print "%s: links: %d/%d rechts: %d/%d" % (name, left_turn_count, len(intersection_angles), right_turn_count, len(intersection_angles))
        sns.distplot(intersection_angles, bins=20, kde=False, rug=True, ax=ax)
        # sns.plt.bar(intersection_angles, bins=20)
        # ax.set_xlabel(u"Kreuzungswinkel [°]")
        # if i == 0:
        #     ax.set_ylabel("Anzahl der Kreuzungen [-]")
    # sns.plt.show(figure1)

    # Oneways
    # figure2, axes2 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        # ax = axes2[i]
        oneways_entry = [extract_features.float_to_boolean(f) for f in get_array_from_feature(samples, 'oneway_entry')]
        oneways_exit = [extract_features.float_to_boolean(f) for f in get_array_from_feature(samples, 'oneway_exit')]
        oneways = [en and ex for en, ex in zip(oneways_entry, oneways_exit)]

        print "%s: Kreuzungen mit Einbahnstraßen: %d/%d" % (name, oneways.count(True), len(samples))
        # oneway = {True:u"Einbahnstraße", False:u"Gegenverkehrsstraße"}
        # sns.set_style("whitegrid")
        # sns.countplot([oneway[o] for o in oneways], order=[u"Einbahnstraße", u"Gegenverkehrsstraße"], ax=ax)
        # if i == 0:
        #     ax.set_ylabel("Kreuzungen mit entsprechenden Armen")
        # else:
        #     ax.set_ylabel("")
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
        # ax.set_xlabel(u"Erlaubte Höchstgeschwindigkeit [km/h]")
        # if i == 0:
        #     ax.set_ylabel("Anzahl der Kreuzungsarme [-]")
        ax.set_ylabel("")
    # sns.plt.show(figure3)

    # Lanes
    figure4, axes4 = sns.plt.subplots(1, 2, sharey=True)
    for i, (name, samples) in enumerate(dataset_samples):
        ax = axes4[i]
        lanes = list(get_array_from_feature(samples, 'lane_count_entry')) + list(get_array_from_feature(samples, 'lane_count_exit'))
        lanes_labels = [label_int(ms) for ms in lanes]
        lanes_plot = sns.countplot(lanes_labels, order=["n/a", "1", "2", "4"], ax=ax)
        # ax.set_xlabel(u"Fahrstreifenanzahl")
        # if i == 0:
        #     ax.set_ylabel("Anzahl der Kreuzungsarme [-]")
        # sns.plt.title("KITTI + Karlsruhe")
        ax.set_ylabel("")
    sns.plt.show(figure4)
    # Track curvatures
    figure5, axes5 = sns.plt.subplots(1, 2)
    for i, (name, samples) in enumerate(dataset_samples):
        l_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] >= 0.]
        r_samples = [s for s in samples if s['X'][_feature_types.index('intersection_angle')] < 0.]
        ax = axes5[i]
        ax.hold(True)
        l_handle = plot_sample_intersection_curvature(l_samples, ax=ax, color=(0,0,1,0.5))
        r_handle = plot_sample_intersection_curvature(r_samples, ax=ax, color=(1,0,0,0.5))
        ax.legend([l_handle, r_handle], ['Linksabb.', 'Rechtsabb.'])
        ax.set_xlim([-30,30])
        ax.set_ylim([-15,15])
        ax.set_ylabel("")
    sns.plt.show(figure5)

    figure6, axes6 = sns.plt.subplots(2, 2, sharey=True, sharex=True)
    for i, (name, samples) in enumerate(dataset_samples):
        speed, dist = zip(*[get_average_speed_and_dist_of_sample(s) for s in samples])
        print "%s: Durchschnittsgeschwindigkeit %d km/h" % (name, np.mean(speed))
        print "%s: Weglänge %.2f m" % (name, np.sum(dist))
        maxspeed_entry = get_array_from_feature(samples, 'maxspeed_entry')
        maxspeed_exit = get_array_from_feature(samples, 'maxspeed_exit')
        vehicle_speed_entry = get_array_from_feature(samples, 'vehicle_speed_entry')
        vehicle_speed_exit = get_array_from_feature(samples, 'vehicle_speed_exit')
        diff_speed_entry = [vs - ms for ms, vs in zip(maxspeed_entry, vehicle_speed_entry) if ms != 0.0]
        diff_speed_exit = [vs - ms for ms, vs in zip(maxspeed_exit, vehicle_speed_exit) if ms != 0.0]
        ax = axes6[0,i]
        sns.distplot(diff_speed_entry, bins=20, kde=False, rug=True, ax=ax)
        ax = axes6[1,i]
        sns.distplot(diff_speed_exit, color="red", bins=20, kde=False, rug=True, ax=ax)
    sns.plt.show(figure6)
