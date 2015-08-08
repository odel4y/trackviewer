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

# Features
# - Straßenwinkel zueinander
# - Geschwindigkeitsbegrenzung am Eingang ausgang
# - Fahrstreifenwahl am Eingang/Ausgang
#   + Angegeben durch Abstand der Fahrstreifenmitte von Mittellinie der Straße in Openstreetmap
# - Ist Eingang oder Ausgang Einbahnstraße?
_features = {
    "intersection_angle": None,
    "maxspeed_entry": None,
    "maxspeed_exit": None,
    "lane_distance_entry": None,
    "lane_distance_exit": None,
    "oneway_entry": None,
    "oneway_exit": None
}

_label = {
    "angles": None,
    "radii": None
}

INT_DIST = 30.0   # distance of the secant construction points from the intersection center [m]
ANGLE_RES = 25   # the angle resolution when sampling the track in polar coordinates with the curve secant centroid as origin
MAX_OSM_TRIES = 3 # the maximum number of tries to download OSM data

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

def get_intersection_angle(entry_line, exit_line):
    """Returns the angle between entry and exit way in radians with parallel ways being a zero angle.
    Only the segments touching the intersection are considered"""
    entry_v = np.array(entry_line.coords[-1]) - np.array(entry_line.coords[-2])
    exit_v = np.array(exit_line.coords[1]) - np.array(exit_line.coords[0])
    normal = np.cross(entry_v, exit_v) # The sign of the angle can be determined
    return np.arccos(np.dot(entry_v, exit_v)/(np.linalg.norm(entry_v) * np.linalg.norm(exit_v))) * normal / np.linalg.norm(normal)

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

def get_track_line(track):
    """Constructs a LineString from the Track"""
    coords = [(x, y) for (x, y, _) in track]
    track_line = LineString(coords)
    return track_line

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

#def get_lane_distance(way_line, track_line, curve_secant):
#    extended_curve_secant = extend_line(curve_secant, 0.1)
#    p1 = way_line.intersection(extended_curve_secant)
#    return track_line.distance(p1)

#def get_lane_distance(w_point, track_line):
#    return track_line.distance(w_point)

def find_closest_intersection(normal, normal_p, track_line):
    """Helper function to handle the different types of intersection"""
    intsec = normal.intersection(track_line)
    if type(intsec) == Point:
        return normal_p.distance(intsec)
    elif type(intsec) == MultiPoint:
        distances = []
        for p in intsec:
            distances.append(normal_p.distance(p))
        return min(distances)
    elif type(intsec) == GeometryCollection and len(intsec) <= 0:
        return None
    else: raise Exception("No valid intersection type")

def get_lane_distance(way_line, p_dist, track_line, normalized=False):
    """Get the distance of the track to the way projected along its normal at p_dist.
    The distance is positive for the right hand and negative for the left hand from the center line."""
    # Construct the normal and its negative counterpart to the line at p_dist
    normal, neg_normal = get_normal_to_line(way_line, p_dist, normalized=normalized)
    normal_p = extended_interpolate(way_line, p_dist, normalized=normalized)
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

def get_normal_to_line(line, dist, normalized=False):
    NORMAL_DX = 0.01 # Distance away from the center point to construct a vector
    if not normalized:
        dist = dist/line.length
    pc = extended_interpolate(line, dist, normalized=True)
    p1 = extended_interpolate(line, dist-NORMAL_DX, normalized=True)
    p2 = extended_interpolate(line, dist+NORMAL_DX, normalized=True)
    if not LineString([p1,p2]).intersects(pc):
        print 'Warning: Normal to line might be inaccurate'
    v1 = np.array([p2.x - p1.x, p2.y - p1.y, 0.0])
    v2 = np.array([0.0, 0.0, 1.0])
    normal = np.cross(v1, v2)
    normal = tuple(normal/np.linalg.norm(normal))[0:2]
    normal_line = LineString([(pc.x, pc.y), (pc.x + normal[0], pc.y + normal[1])])
    neg_normal_line = LineString([(pc.x, pc.y), (pc.x - normal[0], pc.y - normal[1])])
    return normal_line, neg_normal_line

def sample_track(curve_secant, track_line, intersection_angle):
    """Sample the track's distance to the centroid of the curve_secant at constant angle steps.
    Returns polar coordinates"""
    origin = curve_secant.interpolate(0.5, normalized=True)
    half_curve_secant = LineString([origin,\
                                    curve_secant.interpolate(0.0, normalized=True)])
    extended_ruler = extend_line(half_curve_secant, 100.0, direction="forward")
    angles = []
    radii = []
    angle_steps = np.linspace(0.0, np.pi, ANGLE_RES)
    for angle in np.nditer(angle_steps):
        # depending on whether it is a right or a left turn the ruler has to rotate in different directions
        rotated_ruler = affinity.rotate(extended_ruler, copysign(angle,intersection_angle), origin=origin, use_radians=True)
        r = find_closest_intersection(rotated_ruler, origin, track_line)
        if r == None: raise Exception("Sampling the track failed")
        angles.append(float(angle))
        radii.append(float(r))
    return angles, radii

def plot_intersection(entry_line, exit_line, track_line, curve_secant):
    def plot_line(color='b', *line):
        for l in line:
            coords = list(l.coords)
            x,y = zip(*coords)
            plt.plot(x,y, color+'-')
    normal_en, neg_normal_en = get_normal_to_line(entry_line, entry_line.length-INT_DIST, normalized=False)
    normal_ex, neg_normal_ex = get_normal_to_line(exit_line, INT_DIST, normalized=False)
    fig = plt.figure()
    plt.hold(True)
    plt.axis('equal')
    plot_line('b', entry_line, exit_line)
    plot_line('m', normal_en, normal_ex)
    plot_line('g', neg_normal_en, neg_normal_ex)
    plot_line('r', track_line)
    plot_line('k', curve_secant)
    plt.show(block=False)

def plot_sampled_track(track_points):
    fig = plt.figure()
    plt.hold(True)
    x, y = zip(*track_points)
    plt.plot(x,y,'b.-')
    plt.show()

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        try:
            print 'Processing %s' % (fne)
            with open(fn, 'r') as f:
                int_sit = pickle.load(f)
            print 'Downloading OSM...'
            osm = transform_osm_to_cartesian(get_osm_data(int_sit))
            print 'Done.'
            int_sit["track"] = transform_track_to_cartesian(int_sit["track"])
            entry_way = get_element_by_id(osm, int_sit["entry_way"])
            exit_way = get_element_by_id(osm, int_sit["exit_way"])
            entry_line, exit_line = get_way_lines(int_sit, osm)
            curve_secant = get_curve_secant_line(entry_line, exit_line)
            track_line = get_track_line(int_sit["track"])
            features = copy.deepcopy(_features)
            features["intersection_angle"] = float(get_intersection_angle(entry_line, exit_line))
            features["maxspeed_entry"] = float(get_maxspeed(entry_way))
            features["maxspeed_exit"] = float(get_maxspeed(exit_way))
            features["oneway_entry"] = get_oneway(entry_way)
            features["oneway_exit"] = get_oneway(exit_way)
            features["lane_distance_entry"] = float(get_lane_distance(entry_line, entry_line.length-INT_DIST, track_line))
            features["lane_distance_exit"] = float(get_lane_distance(exit_line, INT_DIST, track_line))
            label = copy.deepcopy(_label)
            angles, radii = sample_track(curve_secant, track_line, features["intersection_angle"])
            label["angles"] = angles
            label["radii"] = radii
            import json
            text = json.dumps(features, sort_keys=True, indent=4)
            print text
            fc = (features, label)
            with open(os.path.join(fp, '..', 'training_data', fne), 'wb') as f:
                pickle.dump(fc, f)
                print 'Wrote', fne
            #plot_intersection(entry_line, exit_line, track_line, curve_secant)
            #plot_sampled_track(label["track_points"])
        except Exception as e:
            print '################'
            print '################'
            print e
            print 'Stepping to next file...'
            print '################'
            print '################'
