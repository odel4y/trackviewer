#!/usr/bin/python
#coding:utf-8
from __future__ import division
import sys
import os.path
import pickle
import overpass
import pyproj
import copy
# import plot_helper
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely import affinity
from math import copysign
import numpy as np
import copy
import pdb
from constants import INT_DIST, SAMPLE_RESOLUTION, MAX_OSM_TRIES, LANE_WIDTH
from datetime import datetime, timedelta

_feature_types = [
    "intersection_angle",                       # Angle between entry and exit way
    "maxspeed_entry",                           # Allowed maximum speed on entry way
    "maxspeed_exit",                            # Allowed maximum speed on exit way
    "lane_distance_entry_exact",                # Distance of track line to curve secant center point at 0 degree angle
    "lane_distance_exit_exact",                 # Distance of track line to curve secant center point at 180 degree angle
    "lane_distance_entry_lane_center",          # Distance of lane center line to curve secant ceter point at 0 degree angle
    "lane_distance_exit_lane_center",           # Distance of lane center line to curve secant ceter point at 180 degree angle
    "lane_distance_entry_projected_normal",     # Distance of track line to entry way at INT_DIST projected along normal
    "lane_distance_exit_projected_normal",      # Distance of track line to exit way at INT_DIST projected along normal
    "oneway_entry",                             # Is entry way a oneway street?
    "oneway_exit",                              # Is exit way a oneway street?
    "curvature_entry",                          # Curvature of entry way over INT_DIST
    "curvature_exit",                           # Curvature of exit way over INT_DIST
    "vehicle_speed_entry",                      # Measured vehicle speed on entry way at INT_DIST
    "vehicle_speed_exit",                       # Measured vehicle speed on exit way at INT_DIST
    "bicycle_designated_entry",                 # Is there a designated bicycle way in the entry street?
    "bicycle_designated_exit",                  # Is there a designated bicycle way in the exit street?
    "lane_count_entry",                         # Total number of lanes in entry way
    "lane_count_exit",                          # Total number of lanes in exit way
    "has_right_of_way",                         # Does the vehicle with the respective manoeuver have right of way at the intersection?
    "curve_secant_dist"                         # Shortest distance from curve secant to intersection center
]
_features = {name: None for name in _feature_types}

_label = {
    "radii": None,              # Radii measured from middle of curve_secant
    "distances": None           # Distance measured from way_line along half_angle_vec
}

_sample = {
    'geometry': {
        'entry_line': None,
        'exit_line': None,
        'curve_secant': None,
        'track_line': None,
        'half_angle_line': None
    },
    'X': None,                          # Feature vector
    'y': None,                          # Label vector with selected method
    'label': {
        'y_radii': None,                # Label vector with sampled data measured as polar coordinates from middle of curve_secant
        'y_distances': None,            # Label vector with sampled data along half_angle_vec
        'selected_method': 'y_radii'    # Selected label method
    },
    'pickled_filename': None            # Filename of prepared pickle file for later lookup
}

_regarded_highways = ["motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "service", "living_street", "track",
            "road"]

class SampleTaggingError(Exception):
    pass

class SampleError(Exception):
    pass

class MaxspeedMissingError(Exception):
    pass

class NoIntersectionError(Exception):
    pass

class ElementMissingInOSMError(Exception):
    pass

def get_osm_data(int_sit):
    tries = 0
    while tries < 3:
        try:
            tries += 1
            api = overpass.API()
            # Returns entry and exit way as well as child nodes
            # search_str = '(way(%d);way(%d););(._;>>;);out;' % \
            #                 (int_sit["entry_way"], int_sit["exit_way"])
            # Returns all participating ways in intersection as well as child nodes
            search_str = '(node(%d);way(bn););(._;>;);out;' % (int_sit["intersection_node"])
            result = api.Get(search_str)
            return result["elements"]
        except Exception as e:
            print e
            print 'Retrying OSM Download...'

def transform_osm_to_cartesian(osm):
    for el in osm:
        if el["type"] == "node":
            el["x"], el["y"] = transform_to_cartesian(el["lon"], el["lat"])
    return osm

def transform_track_to_cartesian(track):
    new_track = []
    for lon, lat, time in track:
        x, y = transform_to_cartesian(lon, lat)
        new_track.append((x, y, time))
    return new_track

def transform_to_cartesian(lon, lat):
    """Transform longitude/latitude values to a cartesian system"""
    in_proj = pyproj.Proj(init='epsg:4326')    # Längen-/Breitengrad
    out_proj = pyproj.Proj(init='epsg:3857')   # Kartesische Koordinaten
    x,y = pyproj.transform(in_proj, out_proj, lon, lat)
    return x,y

def transform_to_lon_lat(x, y):
    """Transform to longitude/latitude values from a cartesian system"""
    in_proj = pyproj.Proj(init='epsg:3857')   # Kartesische Koordinaten
    out_proj = pyproj.Proj(init='epsg:4326')    # Längen-/Breitengrad
    lon,lat = pyproj.transform(in_proj, out_proj, x, y)
    return lon,lat

def get_element_by_id(osm, el_id):
    """Returns the element with el_id from osm"""
    osm_iter = iter(osm)
    this_el = osm_iter.next()
    while this_el["id"] != el_id:
        try:
            this_el = osm_iter.next()
        except StopIteration:
            raise ElementMissingInOSMError("Element with ID: %d was not found in OSM data" % el_id)
    return this_el

def get_line_string_from_node_ids(osm, nodes):
    """Constructs a LineString from a list of Node IDs"""
    coords = []
    for node_id in nodes:
        this_node = get_element_by_id(osm, node_id)
        coords.append((this_node["x"], this_node["y"]))
    return LineString(coords)

def get_node_ids_from_way(osm, way, start_node_id=None, end_node_id=None, undershoot=False, overshoot=False):
    """Get a list of node ids from a way starting at start_node_id or the start of
    the way (undershoot==True) and ending at end_node_id included or until the end of the way (overshoot==True)"""
    way_node_ids = way["nodes"]
    if start_node_id != None and end_node_id != None:
        start_ind = way_node_ids.index(start_node_id)
        end_ind = way_node_ids.index(end_node_id)
        if start_ind < end_ind:
            if undershoot: start_ind = 0
            if overshoot: end_ind = len(way_node_ids)-1
            return way_node_ids[start_ind:end_ind+1]
        if start_ind > end_ind:
            if undershoot: start_ind = len(way_node_ids)-1
            if overshoot: end_ind = 0
            return reversed(way_node_ids[end_ind:start_ind+1])
    else:
        return way_node_ids

def way_is_pointing_towards_intersection(way, int_sit):
    """Determine if OSM way's direction is towards the intersection center.
    This method only makes sense for ways that end at the intersection"""
    intersection_node_id = int_sit["intersection_node"]
    return way["nodes"].index(intersection_node_id) > 0

def way_crosses_intersection(way, int_sit):
    """Determine if a way crosses the intersection (does not end at it)"""
    intersection_node_index = way["nodes"].index(int_sit["intersection_node"])
    return 0 < intersection_node_index < (len(way["nodes"]) - 1)

def split_way_at(way, index):
    """Splits way at an index so that the two parts both contain the node with index"""
    part1 = copy.deepcopy(way)
    part2 = copy.deepcopy(way)
    del part1["nodes"][index+1:]
    del part2["nodes"][:index]
    return (part1, part2)

def split_ways_at_intersection(ways, int_sit):
    """Split ways that are crossing the intersection into two separate parts"""
    split_ways = []
    for way in ways:
        if way_crosses_intersection(way, int_sit):
            intersection_node_index = way["nodes"].index(int_sit["intersection_node"])
            split_ways.extend(split_way_at(way, intersection_node_index))
        else:
            split_ways.append(way)
    return split_ways

def delete_duplicate_ways(ways):
    delete_indices = []
    for i, cmp_way in enumerate(ways):
        for j in range(i+1, len(ways)):
            if cmp_way["nodes"] == ways[j]["nodes"]:
                delete_indices.append(i)
    return [way for i, way in enumerate(ways) if i not in delete_indices]

def get_intersection_ways(int_sit, osm):
    """Find all suitable ways in OSM data and split them at the intersection
    if necessary. Then separate the entry_way and exit_way from the other ways
    and save into dictionary"""
    def is_suitable_way(el):
        return  el["type"] == "way" and \
                "highway" in el["tags"] and \
                el["tags"]["highway"] in _regarded_highways
    intersection_ways = [el for el in osm if is_suitable_way(el)]
    intersection_ways = delete_duplicate_ways(intersection_ways)
    # Split ways at the intersection center
    intersection_ways = split_ways_at_intersection(intersection_ways, int_sit)
    # Filter out and remove entry and exit ways -> potentially only removing the part that enters/exits the intersection
    try:
        entry_way, = [way for way in intersection_ways if way["id"] == int_sit["entry_way"] and int_sit["entry_way_node"] in way["nodes"]]
        exit_way, = [way for way in intersection_ways if way["id"] == int_sit["exit_way"] and int_sit["exit_way_node"] in way["nodes"]]
    except ValueError:
        raise SampleTaggingError("Not exactly 1 way has been found for entry or exit")
    intersection_ways.remove(entry_way)
    intersection_ways.remove(exit_way)
    return {
        "entry_way": entry_way,
        "exit_way": exit_way,
        "other_ways": intersection_ways
    }

def get_intersection_way_lines(ways, int_sit, osm):
    """Get a dictionary of LineString for the respective ways. All LineStrings
    are directed away from the intersection except for the entry_way"""
    lines = {}
    if way_is_pointing_towards_intersection(ways["entry_way"], int_sit):
        lines["entry_way"] = get_line_string_from_node_ids(osm, ways["entry_way"]["nodes"])
    else:
        lines["entry_way"] = get_line_string_from_node_ids(osm, reversed(ways["entry_way"]["nodes"]))
    if not way_is_pointing_towards_intersection(ways["exit_way"], int_sit):
        lines["exit_way"] = get_line_string_from_node_ids(osm, ways["exit_way"]["nodes"])
    else:
        lines["exit_way"] = get_line_string_from_node_ids(osm, reversed(ways["exit_way"]["nodes"]))
    other_lines = []
    for way in ways["other_ways"]:
        if not way_is_pointing_towards_intersection(way, int_sit):
            other_lines.append(get_line_string_from_node_ids(osm, way["nodes"]))
        else:
            other_lines.append(get_line_string_from_node_ids(osm, reversed(way["nodes"])))
    lines["other_ways"] = other_lines
    return lines

# def get_way_lines(int_sit, osm):
#     """Return LineStrings for the actual entry and exit way separated by the intersection node.
#     The resulting entry way faces towards the intersection and the exit way faces away from the intersection."""
#     # Get all ways from osm data that are arms of the intersection
#     intersection_ways = get_intersection_ways(osm)
#     # Split ways at the intersection center
#     intersection_ways = split_ways_at_intersection(intersection_ways, int_sit)
#     # Filter out and remove entry and exit ways -> potentially only removing the part that enters/exits the intersection
#     entry_way, = [way for way in intersection_ways if way["id"] == int_sit["entry_way"] and int_sit["entry_way_node"] in way["nodes"]]
#     exit_way, = [way for way in intersection_ways if way["id"] == int_sit["exit_way"] and int_sit["exit_way_node"] in way["nodes"]]
#     intersection_ways.remove(entry_way)
#     intersection_ways.remove(exit_way)
#     # Reverse ways if necessary for standardized orientation and make LineStrings
#     if way_is_pointing_towards_intersection(entry_way, int_sit):
#         entry_way_line_string = get_line_string_from_node_ids(osm, entry_way["nodes"])
#     else:
#         entry_way_line_string = get_line_string_from_node_ids(osm, reversed(entry_way["nodes"]))
#     if not way_is_pointing_towards_intersection(exit_way, int_sit):
#         exit_way_line_string = get_line_string_from_node_ids(osm, exit_way["nodes"])
#     else:
#         exit_way_line_string = get_line_string_from_node_ids(osm, reversed(exit_way["nodes"]))
#     return (entry_way_line_string, exit_way_line_string)

def get_vec_angle(vec1, vec2):
    """Returns angle in radians between two vectors"""
    normal = np.cross(vec1, vec2) # The sign of the angle can be determined
    angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))) * normal / np.linalg.norm(normal)
    if np.isnan(angle): angle = 0.0 # Account for numeric inaccuracies
    return angle

def xy_vec_to_polar(vec):
    """Transforms a vector with x and y components into a polar one with r and phi components"""
    x, y = vec[0], vec[1]
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, phi

def polar_vec_to_xy(vec):
    """Transforms a polar vector with r and phi components into a cartesian one with x and y components"""
    r, phi = vec[0], vec[1]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def get_angle_between_lines(line1, line2):
    """The same as get_intersection_angle but both lines are pointing away from origin"""
    vec1 = np.array(line1.coords[1]) - np.array(line1.coords[0])
    vec2 = np.array(line2.coords[1]) - np.array(line2.coords[0])
    return get_vec_angle(vec1, vec2)

def get_intersection_angle(entry_line, exit_line):
    """Returns the angle between entry and exit way in radians with parallel ways being a zero angle.
    Only the segments touching the intersection are considered"""
    #TODO: Use the averaged intersection angle over the distance INT_DIST instead?
    entry_v = np.array(entry_line.coords[-1]) - np.array(entry_line.coords[-2])
    exit_v = np.array(exit_line.coords[1]) - np.array(exit_line.coords[0])
    return get_vec_angle(entry_v, exit_v)

def mph_to_kmh(v):
    return 1.609344 * v

def get_maxspeed(way):
    """Get the tagged maxspeed for a way. If no maxspeed is
    tagged try to guess it from the highway tag or adjacent ways"""
    if "maxspeed" in way["tags"]:
        try:
            return float(way["tags"]["maxspeed"])
        except ValueError:
            # Maxspeed is formatted as a string
            # Guess format:
            args = way["tags"]["maxspeed"].split(' ')
            if args[1] == 'mph':
                return mph_to_kmh(float(args[0]))
            elif args[1] == 'kmh':
                return float(args[0])
    else:
        # If no tag maxspeed is found try to guess it
        way_name = way["tags"]["name"]
        way_name = way_name.encode('ascii', 'ignore')
        if way["tags"]["highway"] in ["primary", "secondary", "tertiary"]:
            # It is a larger street so 50km/h as maxspeed is assumable
            print 'Assuming maxspeed 50km/h for %s (ID: %d) (highway=%s)' % (way_name, way["id"], way["tags"]["highway"])
            return 50.0
        elif way["tags"]["highway"] in ["residential", "unclassified"]:
            print 'Assuming maxspeed 30km/h for %s (ID: %d) (highway=%s)' % (way_name, way["id"], way["tags"]["highway"])
            return 30.0
        else:
            raise MaxspeedMissingError(u"No maxspeed could be found for %s (ID: %d)" % (way_name, way["id"]))

def get_oneway(way):
    """Determine whether way is a oneway street"""
    if "oneway" in way["tags"]:
        return way["tags"]["oneway"] == "yes"
    else:
        return False

def get_bicycle_designated(way):
    """Determine whether way has a designated bycicle path"""
    if "bicycle" in way["tags"]:
        return way["tags"]["bicycle"] == "designated"
    else:
        return False

def get_lane_count(way):
    """Get total number of lanes on this way. Default is 2"""
    # TODO: Does not support lanes:forward/backward tag
    if "lanes" in way["tags"]:
        lanes = way["tags"]["lanes"]
        if lanes < 2 and get_oneway(way) == False:
            # A street that is not oneway must have at least 2 lanes
            print "Correcting lanes to 2 because street is not oneway"
            lanes = 2
        return lanes
    else:
        if get_oneway(way):
            print "Guessing 1 lane -> oneway"
            return 1
        else:
            print "Guessing 2 lanes"
            return 2

def get_curve_secant_dist(entry_line, curve_secant):
    """Calculate shortest distance from curve secant to intersection center"""
    intersection_center_p = entry_line.interpolate(1.0, normalized=True)
    return curve_secant.distance(intersection_center_p)

def way_has_priority_by_tags(way):
    """Determine if a certain way is prioritized by its tags"""
    if "highway" in way and way["tags"]["highway"] in ["primary", "secondary", "tertiary"]:
        # Entry way is a through road ("Durchgangsstraße")
        # It thus has priority
        return True
    elif "priority_road" in way["tags"] and way["tags"]["priority_road"] == "designated":
        # Designated priority road -> almost never occurs in OSM data
        return True
    else:
        return False

def get_has_right_of_way(ways, way_lines, int_sit):
    # TODO: Does not represent intersections with signals
    # TODO: Does not consider priority signs
    if way_has_priority_by_tags(ways["entry_way"]):
        print "Entry way has the right-of-way by tags [has_right_of_way=True]"
        return True
    # Find out the intersection angle between entry_way and exit_way
    target_intersection_angle = get_intersection_angle(way_lines["entry_way"], way_lines["exit_way"])
    # Find ways that have a lower intersection_angle and thus are to the right and
    # might have to be prioritized
    ways_to_the_right = []
    for way, way_line in zip(ways["other_ways"], way_lines["other_ways"]):
        this_intersection_angle = get_intersection_angle(way_lines["entry_way"], way_line)
        if this_intersection_angle < target_intersection_angle:
            ways_to_the_right.append(way)
    # Determine if the respective way to the right has to be prioritized
    for way in ways_to_the_right:
        # Is the way pointing at the intersection? -> Cars definitely enter from that way
        # If it does not point at the intersection check if it is a oneway
        if way_is_pointing_towards_intersection(way, int_sit) or get_oneway(way)==False:
            print "Ways from the right have priority [has_right_of_way=False]"
            return False
    # Any way has priority and entry_way does not?
    for way in ways["other_ways"]:
        if way_has_priority_by_tags(way):
            if way_is_pointing_towards_intersection(way) or get_oneway(way)==False:
                print "Other ways are explicitly prioritized [has_right_of_way=False]"
                return False
    # By default assume to have right-of-way
    print "Default right-of-way [has_right_of_way=True]"
    return True

def find_nearest_coord_index(line, ref_p):
    """Returns the index of the least distant coordinate of a LineString line
    to ref_p"""
    min_dist = None
    min_i = None
    for i, (x,y) in enumerate(line.coords):
        dist = ref_p.distance(Point(x,y))
        if min_dist == None or dist < min_dist:
            min_dist = dist
            min_i = i
    return min_i

def get_vehicle_speed(way_line, dist, track):
    """Returns the measured vehicle speed in km/h at dist of way_line
    with distance INT_DIST from intersection center"""
    track_line = LineString([(x,y) for (x,y,_) in track])
    dist_p = extended_interpolate(way_line, dist)
    normal = extend_line(get_normal_to_line(way_line, dist), 100.0, direction="both")
    track_p = find_closest_intersection(normal, dist_p, track_line)
    if track_p == None:
        print "Could not find a track point normal to lane for speed measurement. Taking the closest one"
        # Just take the start or end point instead
        track_p = track_line.interpolate(track_line.project(dist_p))
    track_i = find_nearest_coord_index(track_line, track_p)
    if track_i < len(track_line.coords)-1-5:
        track_i2 = track_i+5
    else:
        # If track_p is last point in track_line take a point before that instead
        track_i2 = track_i-5
    track_p2 = Point(track_line.coords[track_i2])
    time_delta = track[track_i2][2] - track[track_i][2]
    time_sec_delta = time_delta.total_seconds()
    dist = track_line.project(Point(track_line.coords[track_i])) - track_line.project(track_p2)
    dist = track_p.distance(track_p2)
    return abs(dist/time_sec_delta*3.6)

def upsample_line(line, times):
    """Simply interpolate more points in a LineString to have sample_rate times coordinates"""
    dists = [line.project(Point(c)) for c in line.coords[:]]
    new_dists = []
    for i, el in enumerate(dists[:-1]):
        for j in range(times):
            dist = el + float(j)*float(dists[i+1]-dists[i])/float(times)
            new_dists.append(dist)
    new_dists.append(dists[-1])
    new_coords = [line.interpolate(d) for d in new_dists]
    return LineString(new_coords)

def split_line(line, dist, normalized=False):
    """Split LineString into two LineStrings at dist"""
    split_p = line.interpolate(dist, normalized=normalized)
    near_i = find_nearest_coord_index(line, split_p)
    near_p = Point(line.coords[near_i])
    dist_diff = dist - line.project(near_p, normalized=normalized)
    if dist_diff >= 0: split_i = near_i + 1 # nearest point is before split_p
    else: split_i = near_i  # nearest point is behind split_p
    line1 = LineString(line.coords[:split_i] + split_p.coords[:])
    line2 = LineString(split_p.coords[:] + line.coords[split_i:])
    return line1, line2

def join_lines(line1, line2):
    """Join two lines that touch at the end of line1 and at beginning of line2"""
    if line1.coords[-1] != line2.coords[0]:
        raise Exception("Lines do not touch")
    coords = line1.coords[:] + line2.coords[1:]
    return LineString(coords)

def extend_line(line, dist, direction="both"):
    """Extends a LineString on both ends for length dist"""
    start_c, end_c = [], []
    if direction in ["both", "backward"]:
        # coordinate of extending line segment at start
        slope_vec = np.array(line.coords[0]) - np.array(line.coords[1])
        norm_slope_vec = slope_vec / np.linalg.norm(slope_vec)
        start_c = [tuple(norm_slope_vec * dist + np.array(line.coords[0]))]
    if direction in ["both", "forward"]:
        # coordinate of extending line segment at end
        slope_vec = np.array(line.coords[-1]) - np.array(line.coords[-2])
        norm_slope_vec = slope_vec / np.linalg.norm(slope_vec)
        end_c = [tuple(norm_slope_vec * dist + np.array(line.coords[-1]))]
        # new LineString is composed of new start and end parts plus the existing one
    if direction not in ["forward", "backward", "both"]:
        raise ValueError("Illegal argument for direction in extend_line")
    return LineString(start_c + list(line.coords) + end_c)

def extended_interpolate(line, dist, normalized=False):
    """Acts like the normal interpolate method except when the distance exceeds
    the object's length. Then it first extends the line and then interpolates"""
    if normalized:
        dist = dist*line.length
    if dist > line.length:
        exceeding_dist = dist - line.length
        extended_line = extend_line(line, exceeding_dist, direction="forward")
        return extended_line.interpolate(dist)
    elif dist < 0.0:
        exceeding_dist = abs(dist)
        extended_line = extend_line(line, exceeding_dist, direction="backward")
        return extended_line.interpolate(0.)
    else:
        return line.interpolate(dist)

def get_curve_secant_line(entry_line, exit_line):
    p1 = extended_interpolate(entry_line, entry_line.length - INT_DIST)
    p2 = extended_interpolate(exit_line, INT_DIST)
    curve_secant = LineString([p1, p2])
    #curve_secant_mid = curve_secant.interpolate(0.5, normalized=True)
    return curve_secant

def find_closest_intersection(line, line_p, track_line):
    """Helper function to handle the different types of intersection and
    find the closest intersection if there are more than one"""
    intsec = line.intersection(track_line)
    if type(intsec) == Point:
        return intsec
    elif type(intsec) == MultiPoint:
        distances = [line_p.distance(p) for p in intsec]
        min_index = distances.index(min(distances))
        return intsec[min_index]
    elif type(intsec) == GeometryCollection and len(intsec) <= 0:
        return None
    else: raise Exception("No valid intersection type")

def get_lane_distance_exact(curve_secant, track_line):
    """Get the distance of the track to the centre point of the curve secant at 0
    and 180 degrees angle"""
    origin_p = curve_secant.interpolate(0.5, normalized=True)
    secant_start_p = curve_secant.interpolate(0.0, normalized=True)
    secant_end_p = curve_secant.interpolate(1.0, normalized=True)
    extended_secant_entry = extend_line(LineString([origin_p, secant_start_p]), 100.0, direction="forward")
    extended_secant_exit = extend_line(LineString([origin_p, secant_end_p]), 100.0, direction="forward")
    track_entry_p = find_closest_intersection(extended_secant_entry, origin_p, track_line)
    lane_distance_entry = origin_p.distance(track_entry_p)
    track_exit_p = find_closest_intersection(extended_secant_exit, origin_p, track_line)
    lane_distance_exit = origin_p.distance(track_exit_p)
    return lane_distance_entry, lane_distance_exit

def get_lane_distance_lane_center(entry_line, exit_line, curve_secant):
    """Get the distance of the lane center currently driven on to the centre point
    """
    # Find center of lanes and extend the lines to be sure to intersect with curve secant
    lane_center_line_entry = extend_line(entry_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="backward")
    lane_center_line_exit = extend_line(exit_line.parallel_offset(LANE_WIDTH/2, side='right'), 100.0, direction="forward")
    origin_p = curve_secant.interpolate(0.5, normalized=True)
    secant_start_p = curve_secant.interpolate(0.0, normalized=True)
    secant_end_p = curve_secant.interpolate(1.0, normalized=True)
    extended_secant_entry = extend_line(LineString([origin_p, secant_start_p]), 100.0, direction="forward")
    extended_secant_exit = extend_line(LineString([origin_p, secant_end_p]), 100.0, direction="forward")
    lane_entry_p = find_closest_intersection(extended_secant_entry, origin_p, lane_center_line_entry)
    lane_distance_entry = origin_p.distance(lane_entry_p)
    lane_exit_p = find_closest_intersection(extended_secant_exit, origin_p, lane_center_line_exit)
    lane_distance_exit = origin_p.distance(lane_exit_p)
    return lane_distance_entry, lane_distance_exit

def get_lane_distance_projected_normal(way_line, dist, track_line, normalized=False):
    """Get the distance of the track to the way projected along its normal at dist.
    The distance is positive for the right hand and negative for the left hand from the center line."""
    # Construct the normal and its negative counterpart to the line at dist
    normal, neg_normal = get_normal_to_line(way_line, dist, normalized=normalized, direction="both")
    normal_p = extended_interpolate(way_line, dist, normalized=normalized)
    # Extend lines to be sure that they intersect with track line
    normal = extend_line(normal, 100.0, direction="forward")
    neg_normal = extend_line(neg_normal, 100.0, direction="forward")
    pos_normal_p = find_closest_intersection(normal, normal_p, track_line)
    if pos_normal_p != None:
        dist_n = normal_p.distance(pos_normal_p)
    else:
        dist_n = None
    neg_normal_p = find_closest_intersection(neg_normal, normal_p, track_line)
    if neg_normal_p != None:
        dist_nn = normal_p.distance(neg_normal_p)
    else:
        dist_nn = None
    if dist_n != None and dist_nn != None:
        if dist_n <= dist_nn:
            return dist_n
        else:
            return -dist_nn
    if dist_n == dist_nn == None:
        raise NoIntersectionError("No intersection of normals with track found")
    else:
        if dist_n != None: return dist_n
        else: return -dist_nn

# def get_track_dist_at_way_dist(way_line, dist, track_line):
#     """Get the track's projected dist at the point the way_line normal at dist points at"""
#     # Construct the normal and its negative counterpart to the line at dist
#     normal, neg_normal = get_normal_to_line(way_line, dist, normalized=normalized, direction="both")
#     normal_p = extended_interpolate(way_line, dist, normalized=normalized)
#     # Extend lines to be sure that they intersect with track line
#     normal = extend_line(normal, 100.0, direction="forward")
#     neg_normal = extend_line(neg_normal, 100.0, direction="forward")
#     pos_normal_p = find_closest_intersection(normal, normal_p, track_line)

def get_reversed_line(way_line):
    """Reverse the order of the coordinates in a LineString"""
    rev_line = LineString(reversed(way_line.coords))
    return rev_line

def get_total_line_curvature(way_line):
    """Get the curvature of a line over INT_DIST"""
    normal1 = get_normal_to_line(way_line, 0.0)
    normal2 = get_normal_to_line(way_line, INT_DIST)
    vec1 = np.array(normal1.coords[1]) - np.array(normal1.coords[0])
    vec2 = np.array(normal2.coords[1]) - np.array(normal2.coords[0])
    d_angle = get_vec_angle(vec1, vec2)
    return d_angle/INT_DIST

def get_curvature_at(way_line, dist, normalized=False):
    """Get the curvature of a line at dist"""
    if normalized: dist = dist * way_line.length
    measure_interval_len = 2.0      # Interval length on which the curvature is calculated [m]
    normal1 = get_normal_to_line(way_line, dist - measure_interval_len/2.0)
    normal2 = get_normal_to_line(way_line, dist + measure_interval_len/2.0)
    vec1 = np.array(normal1.coords[1]) - np.array(normal1.coords[0])
    vec2 = np.array(normal2.coords[1]) - np.array(normal2.coords[0])
    d_angle = get_vec_angle(vec1, vec2)
    return d_angle/measure_interval_len

def get_line_curvature(way_line, sample_steps=100):
    """Get the curvature of way_line sampled with sample_steps"""
    measure_interval_len = way_line.length / (sample_steps - 1)
    curvature_list = np.zeros((1,sample_steps))

    normal1 = get_normal_to_line(way_line, 0.0 - measure_interval_len/2.0)
    vec1 = np.array(normal1.coords[1]) - np.array(normal1.coords[0])
    for i in range(sample_steps):
        normal2 = get_normal_to_line(way_line, (float(i) + 0.5)*measure_interval_len)
        vec2 = np.array(normal2.coords[1]) - np.array(normal2.coords[0])
        d_angle = get_vec_angle(vec1, vec2)
        curvature_list[0,i] = d_angle/measure_interval_len
        normal1 = normal2
        vec1 = vec2
    return curvature_list

def get_normal_to_line(line, dist, normalized=False, direction="forward"):
    NORMAL_DX = 0.01 # Distance away from the center point to construct a vector
    if not normalized:
        dist = dist/line.length
    pc = extended_interpolate(line, dist, normalized=True)
    p1 = extended_interpolate(line, dist-NORMAL_DX, normalized=True)
    p2 = extended_interpolate(line, dist+NORMAL_DX, normalized=True)
    v1 = np.array([p2.x - p1.x, p2.y - p1.y, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    normal = np.cross(v1, v2)
    normal = tuple(normal/np.linalg.norm(normal))[0:2]
    normal_line = LineString([(pc.x, pc.y), (pc.x + normal[0], pc.y + normal[1])])
    if direction == "forward":
        return normal_line
    neg_normal_line = LineString([(pc.x, pc.y), (pc.x - normal[0], pc.y - normal[1])])
    if direction == "backward":
        return neg_normal_line
    elif direction == "both":
        return normal_line, neg_normal_line
    else:
        raise NotImplementedError('The option direction="%s" is not implemented.' % direction)

def get_offset_point_at_distance(line, dist, parallel_offset):
    """Get a point normal to line at a parallel_offset to the point on line at dist"""
    normal = get_normal_to_line(line, dist)
    return extended_interpolate(normal, parallel_offset)

def sample_line(curve_secant, track_line, intersection_angle):
    """Sample the line's distance to the centroid of the curve_secant at constant angle steps.
    Returns polar coordinates"""
    origin = curve_secant.interpolate(0.5, normalized=True)
    half_curve_secant = LineString([origin,\
                                    curve_secant.interpolate(0.0, normalized=True)])
    extended_ruler = extend_line(half_curve_secant, 100.0, direction="forward")
    radii = []
    angle_steps = np.linspace(0.0, np.pi, SAMPLE_RESOLUTION)
    for angle in np.nditer(angle_steps):
        # depending on whether it is a right or a left turn the ruler has to rotate in different directions
        rotated_ruler = affinity.rotate(extended_ruler, copysign(angle,intersection_angle), origin=origin, use_radians=True)
        r_p = find_closest_intersection(rotated_ruler, origin, track_line)
        if r_p == None: raise SampleError("Sampling the track failed")
        r = origin.distance(r_p)
        radii.append(float(r))
    return radii

def get_half_angle_vec(exit_line, intersection_angle):
    """Get the vector that is pointing at half the intersection angle between entry and exit"""
    exit_v = np.array(exit_line.coords[1]) - np.array(exit_line.coords[0])
    half_angle = (np.pi - np.abs(intersection_angle)) / 2.0
    half_angle_vec = rotate_xy(exit_v, np.sign(intersection_angle) * half_angle, (0,0))
    return half_angle_vec

def set_up_way_line_and_distances(entry_line, exit_line):
    """Constructs the relevant way_line out of entry and exit line and calculates
    the distance steps"""
    # Get only the relevant parts of entry and exit way
    _, l1 = split_line(entry_line, entry_line.length - INT_DIST)
    l2, _ = split_line(exit_line, INT_DIST)
    way_line = join_lines(l1, l2)
    # Evenly place sampling distances along way_line
    distance_steps = np.linspace(0, way_line.length, SAMPLE_RESOLUTION)
    return way_line, distance_steps

def sample_line_along_half_angle_vec(entry_line, exit_line, half_angle_vec, track_line):
    """Sample the track's distance to entry and exit line along half_angle_vec"""
    way_line, line_dists = set_up_way_line_and_distances(entry_line, exit_line)
    sampled_dist = []
    # Get the distance from way_line to track_line at every sampling position
    for ld in line_dists:
        way_line_p = way_line.interpolate(ld)
        # Construct a ruler along half_angle_vec that can be used to measure the distance from way_line to the track
        pos_ruler_coords = [way_line_p.coords[0], tuple(np.array(way_line_p.coords[0]) + half_angle_vec)]
        pos_ruler = extend_line(LineString(pos_ruler_coords), 100.0, direction="forward")
        neg_ruler_coords = [way_line_p.coords[0], tuple(np.array(way_line_p.coords[0]) - half_angle_vec)]
        neg_ruler = extend_line(LineString(neg_ruler_coords), 100.0, direction="forward")
        d_p = find_closest_intersection(pos_ruler, way_line_p, track_line)
        nd_p = find_closest_intersection(neg_ruler, way_line_p, track_line)
        if d_p != None:
            pdist = way_line_p.distance(d_p)
        else:
            pdist = None
        if nd_p != None:
            ndist = - way_line_p.distance(nd_p)
        else:
            ndist = None
        if pdist != None and ndist != None:
            if pdist < np.abs(ndist):
                sampled_dist.append(pdist)
            else:
                sampled_dist.append(ndist)
        elif pdist != None:
            sampled_dist.append(pdist)
        elif ndist != None:
            sampled_dist.append(ndist)
        else:
            raise SampleError("Track failed to be sampled along half_angle_vec")
    return sampled_dist

# def get_projected_distance(p_coords, line_coords, proj_vec_coords, direction="both"):
#         """Get the distance of a point to a line along proj_vec while only measuring forward/backward or both directions"""
#
#         def get_proj_dist_one_p_mult_vec(direction):
#             """Get the projected distance for one point to a line with multiple proj_vecs"""
#             if direction == "forward":
#                 proj_vecs = proj_vec_coords
#             elif direction == "backward":
#                 proj_vecs = -proj_vec_coords
#
#             # Calculate the angles of all line_coords respective to the proj_vec
#             angles = np.zeros(line_coords.shape[0],proj_vec_coords.shape[0])    # Actually cos angle
#             cross_prod = np.zeros(line_coords.shape[0],proj_vec_coords.shape[0])
#             vecs_to_line_coords = line_coords - p_coords
#             vecs_to_line_coords_lengths = np.linalg.norm(vecs_to_line_coords, axis=1)
#             proj_vecs_lengths = np.linalg.norm(proj_vecs, axis=1)
#
#             for j in xrange(angles.shape[0]):
#                 for k in xrange(angles.shape[1]):
#                     angles[j,k] = np.dot(proj_vecs[k], vecs_to_line_coords[j]) / \
#                                 (proj_vecs_lengths[k] * vecs_to_line_coords_lengths[j])
#                     cross_prod[j,k] = np.cross(proj_vecs[k], vecs_to_line_coords[j])
#
#             # Find the indices of the respective minimum angles
#             min_ind = np.argmax(angles, axis=0)
#             neighbor_ind = np.zeros(min_ind.shape)
#             for j in xrange(min_ind.size):
#                 this_sign = np.sign(cross_prod[min_ind[j], j])
#                 try:
#                     next_sign = np.sign(cross_prod[min_ind[j]+1, j])
#                 except IndexError:
#                     next_sign = None
#                 try:
#                     previous_sign = np.sign(cross_prod[min_ind[j]-1, j])
#                 except IndexError:
#                     previous_sign = None
#
#                 if next_sign != None and previous_sign != None:
#                     if next_sign == previous_sign:
#                         raise NoIntersectionError()
#                     elif next_sign != this_sign:
#                         neighbor_ind[j] = min_ind[j]+1
#                     elif previous_sign != this_sign:
#                         neighbor_ind[j] = min_ind[j]-1
#                     else:
#                         raise NoIntersectionError()
#
#                 raise NoIntersectionError("No intersection could be found while measuring projected distance")
        #
        #
        # def get_proj_dist_mult_p_one_vec(direction):
        #     """Get the projected distance for multiple points to a line with one proj_vec"""

def get_projected_distance(p, line, proj_vec, direction="both", ret_line_dist=False):
    """Get the distance of a point to a line along proj_vec while only measuring forward/backward or both directions.
    Optionally also returns the line distance where the proj_vec crosses the line"""
    SEARCH_LENGTH = 1000.0
    pos_dist, neg_dist = None, None
    pos_line_dist, neg_line_dist = None, None
    x, y = p.coords[0]

    # Construct a ruler along proj_vec that can be used to measure the distance from p to line
    if direction in ["forward", "both"]:
        pos_ruler_coords = [(x, y), tuple(np.array((x, y)) + proj_vec)]
        pos_ruler = extend_line(LineString(pos_ruler_coords), SEARCH_LENGTH, direction="forward")
        pos_intsec_p = find_closest_intersection(pos_ruler, p, line)
        if pos_intsec_p != None:
            pos_dist = p.distance(pos_intsec_p)
            if ret_line_dist:
                pos_line_dist = line.project(pos_intsec_p)

    if direction in ["backward", "both"]:
        neg_ruler_coords = [(x, y), tuple(np.array((x, y)) - proj_vec)]
        neg_ruler = extend_line(LineString(neg_ruler_coords), SEARCH_LENGTH, direction="forward")
        neg_intsec_p = find_closest_intersection(neg_ruler, p, line)
        if neg_intsec_p != None:
            neg_dist = - p.distance(neg_intsec_p)
            if ret_line_dist:
                neg_line_dist = line.project(neg_intsec_p)

    def pos():
        if ret_line_dist:
            return pos_dist, pos_line_dist
        else:
            return pos_dist

    def neg():
        if ret_line_dist:
            return neg_dist, neg_line_dist
        else:
            return neg_dist

    if pos_dist != None and neg_dist != None:
        if pos_dist <= np.abs(neg_dist):
            return pos()
        else:
            return neg()
    elif pos_dist != None:
        return pos()
    elif neg_dist != None:
        return neg()
    else:
        raise NoIntersectionError("No intersection of proj_vec with line found")

def get_distances_from_cartesian(X, Y, way_line, half_angle_vec):
    """Transform coordinates from cartesian xy-coordinates into distances system"""
    LineDistances = np.zeros(np.shape(X))
    MeasureDistances = np.zeros(np.shape(X))
    for i in range(LineDistances.size):
        i_arr = np.unravel_index(i, np.shape(LineDistances))
        if type(i_arr) == tuple and len(i_arr) == 1:
            i_arr = i_arr[0]
        x = X[i_arr]
        y = Y[i_arr]

        md, ld = get_projected_distance(Point((x, y)), way_line, -half_angle_vec, ret_line_dist=True)

        LineDistances[i_arr] = ld
        MeasureDistances[i_arr] = md
    return LineDistances, MeasureDistances

def get_cartesian_from_distances(LineDistances, MeasureDistances, way_line, half_angle_vec):
    """Transform coordinates from distances system into cartesian xy-coordinates"""
    X = np.zeros(np.shape(LineDistances))
    Y = np.zeros(np.shape(LineDistances))
    for i in range(X.size):
        i_arr = np.unravel_index(i, np.shape(X))
        if type(i_arr) == tuple and len(i_arr) == 1:
            i_arr = i_arr[0]
        print i_arr
        ld = LineDistances[i_arr]
        md = MeasureDistances[i_arr]

        way_line_p = way_line.interpolate(ld)
        # Construct a ruler along half_angle_vec that can be used to measure the distance from way_line to the track
        pos_ruler_coords = [way_line_p.coords[0], tuple(np.array(way_line_p.coords[0]) + half_angle_vec)]
        pos_ruler = LineString(pos_ruler_coords)
        xy_p = extended_interpolate(pos_ruler, md)
        X[i_arr], Y[i_arr] = xy_p.coords[0]
    return X, Y

def get_predicted_line_along_half_angle_vec(entry_line, exit_line, half_angle_vec, d_pred):
    """Construct a predicted line along half_angle_vec with d_pred"""
    way_line, line_dists = set_up_way_line_and_distances(entry_line, exit_line)
    pred_line_points = []
    # Construct a point at a distance at every sampling position
    # for ld, pd in zip(line_dists, d_pred):
    #     way_line_p = way_line.interpolate(ld)
    #     # Construct a ruler along half_angle_vec that can be used to measure the distance from way_line to the track
    #     pos_ruler_coords = [way_line_p.coords[0], tuple(np.array(way_line_p.coords[0]) + half_angle_vec)]
    #     pos_ruler = LineString(pos_ruler_coords)
    #     pred_line_points.append(extended_interpolate(pos_ruler, pd))
    X, Y = get_cartesian_from_distances(line_dists, d_pred, way_line, half_angle_vec)
    coords = zip(list(X), list(Y))
    return LineString(coords)

def rotate_xy(coords, phi, rot_c):
    """Rotate coords [n x 2] in 2D plane about the rotation center rot_c [1 x 2] with angle (rad)"""
    # Rotation matrix in 2D plane
    R_mat = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]
    ])
    # Shift coordinates to origin
    origin_coords = np.transpose(coords - rot_c)
    rot_origin_coords = np.dot(R_mat, origin_coords)
    # Shift back to rotation center
    return np.transpose(rot_origin_coords) + rot_c

def get_cartesian_from_polar(R, Phi, curve_secant, intersection_angle):
    """Transform arrays of polar coordinates (rad) to cartesian system with curve secant as origin"""
    origin = curve_secant.interpolate(0.5, normalized=True)
    half_curve_secant = LineString([origin,\
                                    curve_secant.interpolate(0.0, normalized=True)])
    X = np.zeros(np.shape(R))
    Y = np.zeros(np.shape(R))

    def get_xy(r, phi):
        # depending on whether it is a right or a left turn the ruler has to rotate in different directions
        rotated_ruler = affinity.rotate(half_curve_secant, phi*np.sign(intersection_angle), origin=origin, use_radians=True)
        p = extended_interpolate(rotated_ruler, r, normalized=False)
        (x, y), = list(p.coords)
        return x, y

    for i_row in range(np.shape(R)[0]):
        try:
            for j_col in range(np.shape(R)[1]):
                X[i_row, j_col], Y[i_row, j_col] = get_xy(R[i_row, j_col], Phi[i_row, j_col])
        except IndexError:
            # One dimensional array
            X[i_row], Y[i_row] = get_xy(R[i_row], Phi[i_row])

    return (X, Y)

def get_predicted_line_radii(curve_secant, radii_pred, intersection_angle):
    """Convert a prediction to cartesian coordinates and represent it as LineString"""
    angles = np.linspace(0., np.pi, len(radii_pred))
    (X, Y) = get_cartesian_from_polar(radii_pred, angles, curve_secant, intersection_angle)
    coords = zip(list(X), list(Y))
    return LineString(coords)

def get_predicted_line(pred, label_method, sample):
    """Returns the prediction as a LineString with the chosen label method"""
    if label_method == 'y_radii':
        return get_predicted_line_radii(sample['geometry']['curve_secant'], pred, sample['X'][_feature_types.index('intersection_angle')])
    elif label_method == 'y_distances':
        half_angle_vec = get_half_angle_vec(sample['geometry']['exit_line'], sample['X'][_feature_types.index('intersection_angle')])
        return get_predicted_line_along_half_angle_vec(sample['geometry']['entry_line'], sample['geometry']['exit_line'], half_angle_vec, pred)

def get_rectified_mse(y_pred, label_method, sample):
    """Returns a somewhat unbiased MSE by measuring the distance from one line
    to another in direction of a normal"""
    # Construct LineString to use geometric methods
    pred_line = get_predicted_line(y_pred, label_method, sample)
    coords_array = np.array(pred_line.coords[:])
    # Calculate the line lengths at each coordinate
    lengths = np.linalg.norm(coords_array[1:,:] - coords_array[:-1,:], axis=1)  # Respective length for each coordinate step
    step_lengths = [0.] + list(np.cumsum(lengths))  # Cumulated lengths for each coordinate step
    distances = []
    for dist in step_lengths:
        try:
            distances.append(get_lane_distance_projected_normal(pred_line, dist, sample['geometry']['track_line']))
        except NoIntersectionError as e:
            print e
            print 'Skipping this coordinate'
            continue
    return np.mean(np.power(np.array(distances), 2))

def get_osm(int_sit):
    print 'Downloading OSM...'
    osm = get_osm_data(int_sit)
    print 'Done.'
    int_sit["track"] = transform_track_to_cartesian(int_sit["track"])
    return transform_osm_to_cartesian(osm)

def get_intersection_geometry(int_sit, osm):
    ways = get_intersection_ways(int_sit, osm)
    way_lines = get_intersection_way_lines(ways, int_sit, osm)
    curve_secant = get_curve_secant_line(way_lines["entry_way"], way_lines["exit_way"])
    track = int_sit["track"]
    return ways, way_lines, curve_secant, track

def boolean_to_float(b):
    if b: return 1.0
    else: return -1.0

def float_to_boolean(f):
    if f == 1.0: return True
    elif f == -1.0: return False
    else:
        raise ValueError("Could not convert float %.1f to boolean")

def get_feature_dict(int_sit, ways, way_lines, curve_secant, track):
    entry_way = ways["entry_way"]
    exit_way = ways["exit_way"]
    other_ways = ways["other_ways"]
    entry_line = way_lines["entry_way"]
    exit_line = way_lines["exit_way"]
    exit_line = way_lines["exit_way"]
    other_lines = way_lines["other_ways"]

    features = copy.deepcopy(_features)
    track_line = LineString([(x, y) for (x,y,_) in track])
    intersection_angle = float(get_intersection_angle(entry_line, exit_line))
    features["intersection_angle"] =                    intersection_angle
    features["maxspeed_entry"] =                        float(get_maxspeed(entry_way))
    features["maxspeed_exit"] =                         float(get_maxspeed(exit_way))
    features["oneway_entry"] =                          boolean_to_float(get_oneway(entry_way))
    features["oneway_exit"] =                           boolean_to_float(get_oneway(exit_way))
    lane_distance_entry_exact, lane_distance_exit_exact = get_lane_distance_exact(curve_secant, track_line)
    features["lane_distance_entry_exact"] =             float(lane_distance_entry_exact)
    features["lane_distance_exit_exact"] =              float(lane_distance_exit_exact)
    lane_distance_entry_lane_center, lane_distance_exit_lane_center = get_lane_distance_lane_center(entry_line, exit_line, curve_secant)
    features["lane_distance_entry_lane_center"] =       lane_distance_entry_lane_center
    features["lane_distance_exit_lane_center"] =        lane_distance_exit_lane_center
    features["lane_distance_entry_projected_normal"] =  float(get_lane_distance_projected_normal(entry_line, entry_line.length - INT_DIST, track_line))
    features["lane_distance_exit_projected_normal"] =   float(get_lane_distance_projected_normal(exit_line, INT_DIST, track_line))
    features["curvature_entry"] =                       float(get_total_line_curvature(get_reversed_line(entry_line)))
    features["curvature_exit"] =                        float(get_total_line_curvature(get_reversed_line(exit_line)))
    vehicle_speed_entry = get_vehicle_speed(entry_line, entry_line.length - INT_DIST, track)
    vehicle_speed_exit = get_vehicle_speed(exit_line, INT_DIST, track)
    features["vehicle_speed_entry"] =                   float(vehicle_speed_entry)
    features["vehicle_speed_exit"] =                    float(vehicle_speed_exit)
    features["bicycle_designated_entry"] =              boolean_to_float(get_bicycle_designated(entry_way))
    features["bicycle_designated_exit"] =               boolean_to_float(get_bicycle_designated(exit_way))
    features["lane_count_entry"] =                      float(get_lane_count(entry_way))
    features["lane_count_exit"] =                       float(get_lane_count(exit_way))
    features["has_right_of_way"] =                      boolean_to_float(get_has_right_of_way(ways, way_lines, int_sit))
    features["curve_secant_dist"] =                     float(get_curve_secant_dist(entry_line, curve_secant))
    label = copy.deepcopy(_label)
    radii = sample_line(curve_secant, track_line, features["intersection_angle"])
    half_angle_vec = get_half_angle_vec(exit_line, intersection_angle)
    distances = sample_line_along_half_angle_vec(entry_line, exit_line, half_angle_vec, track_line)
    label["radii"] = radii
    label["distances"] = distances
    return features, label

def convert_to_array(features, label):
    """Convert features to a number and put them in a python list"""
    feature_list = [features[feature_name] for feature_name in _feature_types]
    label_list = {
        "radii": np.array(label["radii"]),
        "distances": np.array(label["distances"])
    }
    return np.array(feature_list), label_list

def get_matrices_from_samples(samples):
    """Get feature and label matrices from samples list"""
    X = np.zeros((len(samples),len(samples[0]['X'])))
    y = np.zeros((len(samples),len(samples[0]['y'])))
    for i, s in enumerate(samples):
        X[i] = np.array(s['X'])
        y[i] = np.array(s['y'])
    return X, y

def get_samples_from_matrices(X, y, samples):
    """Update feature and label matrices in samples list"""
    for i, s in enumerate(samples):
        s['X'] = np.array(X[i])
        s['y'] = np.array(y[i])
    return samples

def select_label_method(samples, label_method="y_radii"):
    """If a different label variant is needed overwrite the standard one"""
    for sample in samples:
        sample['y'] = sample['label'][label_method]
        sample['label']['selected_method'] = label_method
    return samples

def create_sample(int_sit, osm, pickled_filename="", output="none"):
    sample = copy.deepcopy(_sample)
    ways, way_lines, curve_secant, track = get_intersection_geometry(int_sit, osm)
    features, label = get_feature_dict(int_sit, ways, way_lines, curve_secant, track)
    track_line = LineString([(x, y) for (x,y,_) in track])

    if output=="console":
        # print features in readable format
        import json
        text = json.dumps(features, sort_keys=True, indent=4)
        print text

    feature_array, label_array = convert_to_array(features, label)
    sample['geometry']['entry_line'] = way_lines["entry_way"]
    sample['geometry']['exit_line'] = way_lines["exit_way"]
    sample['geometry']['curve_secant'] = curve_secant
    sample['geometry']['track_line'] = track_line
    half_angle_vec = get_half_angle_vec(way_lines["exit_way"], features["intersection_angle"])
    half_angle_line = LineString([way_lines["exit_way"].coords[0], tuple(np.array(way_lines["exit_way"].coords[0]) + half_angle_vec)])
    sample['geometry']['half_angle_line'] = half_angle_line
    sample['X'] = feature_array
    sample['y'] = label_array["radii"]
    sample['label']['y_radii'] = label_array["radii"]
    sample['label']['y_distances'] = label_array["distances"]
    sample['label']['selected_method'] = "y_radii"
    sample['pickled_filename'] = pickled_filename

    return sample

if __name__ == "__main__":
    samples = []
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        try:
            print 'Processing %s' % (fne)
            with open(fn, 'r') as f:
                int_sit = pickle.load(f)
            osm = get_osm(int_sit)
            samples.append(create_sample(int_sit, osm, fn, output="console"))
            # plot_helper.plot_intersection(sample)
        except (ValueError, SampleError, MaxspeedMissingError, NoIntersectionError, SampleTaggingError) as e:
            print '################'
            print '################'
            print e
            print 'Stepping to next file...'
            print '################'
            print '################'
    with open(os.path.join(fp, '..', 'training_data', 'samples.pickle'), 'wb') as f:
        print 'Writing database...'
        pickle.dump(samples, f)
