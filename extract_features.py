#!/usr/bin/python
#coding:utf-8
import sys
import os.path
import pickle
import overpass
import pyproj
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely import affinity
from math import copysign
import numpy as np
import copy
import pdb
import matplotlib.pyplot as plt
from constants import INT_DIST, ANGLE_RES, MAX_OSM_TRIES, LANE_WIDTH
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
    "vehicle_speed_exit"                        # Measured vehicle speed on exit way at INT_DIST
]
_features = {name: None for name in _feature_types}

_label = {
    "radii": None
}

_sample = {
    'geometry': {
        'entry_line': None,
        'exit_line': None,
        'track_line': None
    },
    'X': None,                  # Feature vector
    'y': None,                  # Label vector
    'pickled_filename': None    # Filename of prepared pickle file for later lookup
}

def get_osm_data(int_sit):
    tries = 0
    while tries < 3:
        try:
            tries += 1
            api = overpass.API()
            search_str = '(way(%d);way(%d););(._;>>;);out;' % \
                            (int_sit["entry_way"], int_sit["exit_way"])
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

def get_element_by_id(osm, el_id):
    """Returns the element with el_id from osm"""
    osm_iter = iter(osm)
    this_el = osm_iter.next()
    while this_el["id"] != el_id:
        try:
            this_el = osm_iter.next()
        except StopIteration:
            this_el = None
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

def get_way_lines(int_sit, osm):
    """Return LineStrings for the actual entry and exit way separated by the intersection node.
    The resulting entry way faces towards the intersection and the exit way faces away from the intersection."""
    entry_way = get_element_by_id(osm, int_sit["entry_way"])
    exit_way = get_element_by_id(osm, int_sit["exit_way"])
    entry_way_node_ids = get_node_ids_from_way(osm, entry_way, int_sit["entry_way_node"], int_sit["intersection_node"], True, False)
    exit_way_node_ids = get_node_ids_from_way(osm, exit_way, int_sit["intersection_node"], int_sit["exit_way_node"], False, True)
    entry_way_line_string = get_line_string_from_node_ids(osm, entry_way_node_ids)
    exit_way_line_string = get_line_string_from_node_ids(osm, exit_way_node_ids)
    return (entry_way_line_string, exit_way_line_string)

def get_vec_angle(vec1, vec2):
    """Return angle between two vectors"""
    normal = np.cross(vec1, vec2) # The sign of the angle can be determined
    angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))) * normal / np.linalg.norm(normal)
    if np.isnan(angle): angle = 0.0 # Account for numeric inaccuracies
    return angle

def get_intersection_angle(entry_line, exit_line):
    """Returns the angle between entry and exit way in radians with parallel ways being a zero angle.
    Only the segments touching the intersection are considered"""
    #TODO: Use the averaged intersection angle over the distance INT_DIST instead?
    entry_v = np.array(entry_line.coords[-1]) - np.array(entry_line.coords[-2])
    exit_v = np.array(exit_line.coords[1]) - np.array(exit_line.coords[0])
    return get_vec_angle(entry_v, exit_v)

def get_maxspeed(way):
    """Get the tagged maxspeed for a way. If no maxspeed is
    tagged try to guess it from the highway tag or adjacent ways"""
    if "maxspeed" in way["tags"]:
        return float(way["tags"]["maxspeed"])
    else:
        # If no tag maxspeed is found try to guess it
        if way["tags"]["highway"] in ["primary", "secondary"]:
            # It is a larger street so 50km/h as maxspeed is assumable
            print 'Assuming maxspeed 50km/h for %s (ID: %d) (highway=%s)' % (way["tags"]["name"], way["id"], way["tags"]["highway"])
            return 50.0
        elif way["tags"]["highway"] in ["residential"]:
            print 'Assuming maxspeed 30km/h for %s (ID: %d) (highway=%s)' % (way["tags"]["name"], way["id"], way["tags"]["highway"])
            return 30.0
        else:
            print 'No maxspeed could be found for %s (ID: %d)' % (way["tags"]["name"], way["id"])
            return None

def get_oneway(way):
    """Determine whether way is a oneway street"""
    if "oneway" in way["tags"]:
        return way["tags"]["oneway"] == "yes"
    else:
        return False

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
    track_i = find_nearest_coord_index(track_line, track_p)
    track_p2 = track_line.coords[track_i+5]
    time_delta = track[track_i+5][2] - track[track_i][2]
    time_sec_delta = time_delta.total_seconds()
    dist = track_p.distance(track_p2)
    return dist/time_sec_delta*3.6

def extend_line(line, dist, direction="both"):
    """Extends a LineString on both ends for length dist"""
    start_c, end_c = [], []
    if direction in ["both", "backward"]:
        # coordinate of extending line segment at start
        start_slope = np.array(line.coords[0]) - np.array(line.coords[1])
        start_c = [tuple(start_slope * dist + np.array(line.coords[0]))]
    elif direction in ["both", "forward"]:
        # coordinate of extending line segment at end
        end_slope = np.array(line.coords[-1]) - np.array(line.coords[-2])
        end_c = [tuple(end_slope * dist + np.array(line.coords[-1]))]
        # new LineString is composed of new start and end parts plus the existing one
    else:
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
        return extended_line.interpolate(exceeding_dist)
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
        return line_p.distance(intsec)
    elif type(intsec) == MultiPoint:
        distances = []
        for p in intsec:
            distances.append(line_p.distance(p))
        return min(distances)
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
    lane_distance_entry = find_closest_intersection(extended_secant_entry, origin_p, track_line)
    lane_distance_exit = find_closest_intersection(extended_secant_exit, origin_p, track_line)
    return lane_distance_entry, lane_distance_exit

def get_lane_distance_lane_center(entry_line, exit_line, curve_secant):
    """Get the distance of the lane center currently driven on to the centre point
    """
    # Find center of lanes and extend the lines to be sure to intersect with curve secant
    lane_center_entry_line = extend_line(entry_line.parallel_offset(LANE_WIDTH/2), 100.0, direction="backward")
    lane_center_exit_line = extend_line(exit_line.parallel_offset(LANE_WIDTH/2), 100.0, direction="forward")
    origin_p = curve_secant.interpolate(0.5, normalized=True)
    secant_start_p = curve_secant.interpolate(0.0, normalized=True)
    secant_end_p = curve_secant.interpolate(1.0, normalized=True)
    extended_secant_entry = extend_line(LineString([origin_p, secant_start_p]), 100.0, direction="forward")
    extended_secant_exit = extend_line(LineString([origin_p, secant_end_p]), 100.0, direction="forward")
    lane_distance_entry = find_closest_intersection(extended_secant_entry, origin_p, lane_center_entry_line)
    lane_distance_exit = find_closest_intersection(extended_secant_exit, origin_p, lane_center_exit_line)
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
    dist_n = find_closest_intersection(normal, normal_p, track_line)
    dist_nn = find_closest_intersection(neg_normal, normal_p, track_line)
    if dist_n != None and dist_nn != None:
        if dist_n <= dist_nn:
            return dist_n
        else:
            return -dist_nn
    if dist_n == dist_nn == None:
        raise Exception("No intersection of normals with track found")
    else:
        return dist_n or -dist_nn # Return the one that is not None

def get_reversed_line(way_line):
    """Reverse the order of the coordinates in a LineString"""
    rev_line = LineString(reversed(way_line.coords))
    return rev_line

def get_line_curvature(way_line):
    """Get the curvature of a line over INT_DIST"""
    normal1 = get_normal_to_line(way_line, 0.0)
    normal2 = get_normal_to_line(way_line, INT_DIST)
    vec1 = np.array(normal1.coords[1]) - np.array(normal1.coords[0])
    vec2 = np.array(normal2.coords[1]) - np.array(normal2.coords[0])
    d_angle = get_vec_angle(vec1, vec2)
    return d_angle/INT_DIST

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

def sample_line(curve_secant, track_line, intersection_angle):
    """Sample the line's distance to the centroid of the curve_secant at constant angle steps.
    Returns polar coordinates"""
    origin = curve_secant.interpolate(0.5, normalized=True)
    half_curve_secant = LineString([origin,\
                                    curve_secant.interpolate(0.0, normalized=True)])
    extended_ruler = extend_line(half_curve_secant, 100.0, direction="forward")
    radii = []
    angle_steps = np.linspace(0.0, np.pi, ANGLE_RES)
    for angle in np.nditer(angle_steps):
        # depending on whether it is a right or a left turn the ruler has to rotate in different directions
        rotated_ruler = affinity.rotate(extended_ruler, copysign(angle,intersection_angle), origin=origin, use_radians=True)
        r = find_closest_intersection(rotated_ruler, origin, track_line)
        if r == None: raise Exception("Sampling the track failed")
        radii.append(float(r))
    return radii

def get_predicted_line(curve_secant, radii_pred, intersection_angle):
    """Get a prediction for the track along the curve and convert it into a LineString"""
    origin = curve_secant.interpolate(0.5, normalized=True)
    half_curve_secant = LineString([origin,\
                                    curve_secant.interpolate(0.0, normalized=True)])
    angles = np.linspace(0.0, np.pi, len(radii_pred))
    points = []
    for i in xrange(len(radii_pred)):
        # depending on whether it is a right or a left turn the ruler has to rotate in different directions
        rotated_ruler = affinity.rotate(half_curve_secant, copysign(angles[i],intersection_angle), origin=origin, use_radians=True)
        p = extended_interpolate(rotated_ruler, radii_pred[i], normalized=False)
        points.append(p)
    return LineString(points)

def plot_intersection(entry_line, exit_line, curve_secant, track_line, predicted_line=None, comparison_line=None):
    def plot_line(color='b', *line):
        for l in line:
            coords = list(l.coords)
            x,y = zip(*coords)
            plt.plot(x,y, color+'-')
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
    normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False, direction="both")
    normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False, direction="both")
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')
    plot_line('k', entry_line, exit_line)
    plot_line('m', normal_en, normal_ex)
    plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('r', track_line)
    plot_arrows_along_line('r', track_line)
    if predicted_line: plot_line('b', predicted_line)
    if comparison_line: plot_line('m', comparison_line)
    plot_line('k', curve_secant)
    plt.show(block=False)

def plot_sampled_track(label):
    fig = plt.figure()
    plt.hold(True)
    plt.plot(label["angles"],label["radii"],'b.-')
    plt.show()

def get_osm(int_sit):
    print 'Downloading OSM...'
    osm = get_osm_data(int_sit)
    print 'Done.'
    int_sit["track"] = transform_track_to_cartesian(int_sit["track"])
    return transform_osm_to_cartesian(osm)

def get_intersection_geometry(int_sit, osm):
    entry_way = get_element_by_id(osm, int_sit["entry_way"])
    exit_way = get_element_by_id(osm, int_sit["exit_way"])
    entry_line, exit_line = get_way_lines(int_sit, osm)
    curve_secant = get_curve_secant_line(entry_line, exit_line)
    track = int_sit["track"]
    return entry_way, exit_way, entry_line, exit_line, curve_secant, track

def get_feature_dict(int_sit, entry_way, exit_way, entry_line, exit_line, curve_secant, track):
    def convert_boolean(b):
        if b: return 1.0
        else: return -1.0
    features = copy.deepcopy(_features)
    track_line = LineString([(x, y) for (x,y,_) in track])
    features["intersection_angle"] = float(get_intersection_angle(entry_line, exit_line))
    features["maxspeed_entry"] = float(get_maxspeed(entry_way))
    features["maxspeed_exit"] = float(get_maxspeed(exit_way))
    features["oneway_entry"] = convert_boolean(get_oneway(entry_way))
    features["oneway_exit"] = convert_boolean(get_oneway(exit_way))
    lane_distance_entry_exact, lane_distance_exit_exact = get_lane_distance_exact(curve_secant, track_line)
    features["lane_distance_entry_exact"] = float(lane_distance_entry_exact)
    features["lane_distance_exit_exact"] = float(lane_distance_exit_exact)
    lane_distance_entry_lane_center, lane_distance_exit_lane_center = get_lane_distance_lane_center(entry_line, exit_line, curve_secant)
    features["lane_distance_entry_lane_center"] = lane_distance_entry_lane_center
    features["lane_distance_exit_lane_center"] = lane_distance_exit_lane_center
    features["lane_distance_entry_projected_normal"] = float(get_lane_distance_projected_normal(entry_line, entry_line.length - INT_DIST, track_line))
    features["lane_distance_exit_projected_normal"] = float(get_lane_distance_projected_normal(exit_line, INT_DIST, track_line))
    features["curvature_entry"] = float(get_line_curvature(get_reversed_line(entry_line)))
    features["curvature_exit"] = float(get_line_curvature(get_reversed_line(exit_line)))
    vehicle_speed_entry = get_vehicle_speed(entry_line, entry_line.length - INT_DIST, track)
    vehicle_speed_exit = get_vehicle_speed(exit_line, INT_DIST, track)
    features["vehicle_speed_entry"] = float(vehicle_speed_entry)
    features["vehicle_speed_exit"] = float(vehicle_speed_exit)
    label = copy.deepcopy(_label)
    radii = sample_line(curve_secant, track_line, features["intersection_angle"])
    label["radii"] = radii
    return features, label

def convert_to_array(features, label):
    """Convert features to a number and put them in a python list"""
    feature_list = [features[feature_name] for feature_name in _feature_types]
    label_list = label["radii"]
    return feature_list, label_list

if __name__ == "__main__":
    samples = []
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        try:
            print 'Processing %s' % (fne)
            sample = copy.deepcopy(_sample)
            with open(fn, 'r') as f:
                int_sit = pickle.load(f)
            osm = get_osm(int_sit)
            entry_way, exit_way, entry_line, exit_line, curve_secant, track = get_intersection_geometry(int_sit, osm)
            features, label = get_feature_dict(int_sit, entry_way, exit_way, entry_line, exit_line, curve_secant, track)
            # print features in readable format
            import json
            text = json.dumps(features, sort_keys=True, indent=4)
            print text
            feature_list, label_list = convert_to_array(features, label)
            sample['geometry']['entry_line'] = entry_line
            sample['geometry']['exit_line'] = exit_line
            sample['geometry']['track_line'] = track_line
            sample['X'] = feature_list
            sample['y'] = label_list
            sample['pickled_filename'] = fn
            samples.append(sample)
            # plot_intersection(entry_line, exit_line, curve_secant, track_line)
        except Exception as e:
            print '################'
            print '################'
            print e
            print 'Stepping to next file...'
            print '################'
            print '################'
    with open(os.path.join(fp, '..', 'training_data', 'samples.pickle'), 'wb') as f:
        print 'Writing database...'
        pickle.dump(samples, f)
