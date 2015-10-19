#!/usr/bin/python
#coding:utf-8
from __future__ import division
import sys
import os.path
import pickle
from extract_features import *
from extract_features import _feature_types
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
import shapely.affinity
from constants import LANE_WIDTH, INT_DIST

class RectificationError(Exception):
    pass

def rectify_track(track, sample):
    track_line = LineString([(x,y) for x,y,time in track])

    entry_line = sample['geometry']['entry_line']
    exit_line = sample['geometry']['exit_line']
    entry_normal = get_normal_to_line(entry_line, entry_line.length - INT_DIST)
    exit_normal = get_normal_to_line(exit_line, INT_DIST)
    entry_normal_vec = np.array(entry_normal.coords[1]) - np.array(entry_normal.coords[0])
    exit_normal_vec = np.array(exit_normal.coords[1]) - np.array(exit_normal.coords[0])

    oneway_entry = float_to_boolean(sample['X'][_feature_types.index('oneway_entry')])
    oneway_exit = float_to_boolean(sample['X'][_feature_types.index('oneway_exit')])

    # Calculate desired track distances
    if oneway_entry:
        desired_entry_distance = 0.
    else:
        desired_entry_distance = LANE_WIDTH/2
    if oneway_exit:
        desired_exit_distance = 0.
    else:
        desired_exit_distance = LANE_WIDTH/2
    max_position_error = 0.1    # Maximum deviation from desired value at entry and exit [m]
    entry_error = lambda: get_lane_distance_projected_normal(entry_line, entry_line.length - INT_DIST, track_line) - desired_entry_distance
    exit_error = lambda: get_lane_distance_projected_normal(exit_line, INT_DIST, track_line) - desired_exit_distance
    max_iterations = 100    # Rectification failed if it is not reached after max_iterations
    current_iteration = 0
    while abs(entry_error()) > max_position_error or abs(exit_error()) > max_position_error:
        if current_iteration >= max_iterations:
            raise RectificationError("Maximum iterations reached when optimizing track_line")
        if abs(entry_error()) > max_position_error:
            translation_vec = - entry_normal_vec * entry_error()
            track_line = shapely.affinity.translate(track_line, xoff=translation_vec[0], yoff=translation_vec[1])
        if abs(exit_error()) > max_position_error:
            translation_vec = - exit_normal_vec * exit_error()
            track_line = shapely.affinity.translate(track_line, xoff=translation_vec[0], yoff=translation_vec[1])
        current_iteration += 1
    print "Rectification successful."
    proj_coords = [transform_to_lon_lat(x, y) for (x, y) in track_line.coords[:]]
    new_track = [(lon, lat, time) for ((lon,lat), (_,_,time)) in zip(proj_coords, track)]
    return new_track

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        try:
            print 'Processing %s' % (fne)
            with open(fn, 'r') as f:
                int_sit = pickle.load(f)
            osm = get_osm(int_sit)  # Also transforms data into cartesian system
            sample = create_sample(int_sit, osm, fn, output="console")
            int_sit['track'] = rectify_track(int_sit['track'], sample)
            with open(fn+'.rect', 'w') as f:
                pickle.dump(int_sit, f)
        except (ValueError, SampleError, MaxspeedMissingError, NoIntersectionError, SampleTaggingError, RectificationError) as e:
            print e
            print 'Stepping to next file...'
            print '################'