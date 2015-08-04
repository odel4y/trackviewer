#!/usr/bin/python
#coding:utf-8
import sys
import os.path
import pickle
import overpass
import pyproj
from shapely.geometry import LineString
import numpy as np
import copy
import pdb

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
    "track_points": None
}

INT_DIST = 30.0     # distance of the secant construction points from the intersection center [m]

def get_osm_data(int_sit):
    api = overpass.API()
    search_str = '(way(%d);way(%d););(._;>>;);out;' % \
                    (int_sit["entry_way"], int_sit["exit_way"])
    result = api.Get(search_str)
    return result["elements"]

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
    if direction in ["both", "forward"]:
        # coordinate of extending line segment at start
        start_slope = np.array(line.coords[0]) - np.array(line.coords[1])
        start_c = [tuple(start_slope * dist + np.array(line.coords[0]))]
    elif direction in ["both", "backward"]:
        # coordinate of extending line segment at end
        end_slope = np.array(line.coords[-1]) - np.array(line.coords[-2])
        end_c = [tuple(end_slope * dist + np.array(line.coords[-1]))]
        # new LineString is composed of new start and end parts plus the existing one
    else:
        raise ValueError("Illegal argument for direction in extend_line")
    return LineString(start_c + list(line.coords) + end_c)

def get_curve_secant_line(entry_line, exit_line):
    p1 = entry_line.interpolate(entry_line.length - INT_DIST)
    p2 = exit_line.interpolate(INT_DIST)
    curve_secant = LineString([p1, p2])
    #curve_secant_mid = curve_secant.interpolate(0.5, normalized=True)
    return curve_secant

def get_lane_distance(way_line, track_line, curve_secant):
    extended_curve_secant = extend_line(curve_secant, 0.1)
    p1 = way_line.intersection(extended_curve_secant)
    return track_line.distance(p1)

#def get_normal_to_line(line, dist, normalized=False):
#    NORMAL_DX = 0.01 # Distance away from the center point to construct a vector
#    if not normalized:
#        dist = dist/line.length
#    pc = line.interpolate(dist, normalized=True)
#    p1 = line.interpolate(dist-NORMAL_DX, normalized=True)
#    p2 = line.interpolate(dist+NORMAL_DX, normalized=True)
#    if not LineString([p1,p2]).intersects(pc):
#        print 'Warning: Normal to line might be inaccurate'
#    v1 = np.array([p2.x - p1.x, p2.y - p1.y, 0.0])
#    v2 = np.array([0.0, 0.0, 1.0])
#    normal = np.cross(v1, v2)
#    normal = tuple(normal/np.linalg.norm(normal))[0:2]
#    normal_line = LineString([(pc.x, pc.y), (normal[0], normal[1])])
#    return normal_line
    
#def get_distance_along_normal(line1, line2, at_dist, normalized=False):
#    normal_line = get_normal_to_line(line1, at_dist, normalized=normalized)
#    extended_normal_line = extend_line(normal_line, 100.0, direction="forward")
#    p1 = line1.interpolate(at_dist, normalized=normalized)
#    intersection = extended_normal_line.intersection(line2)
#    
#    if len(intersection) > 0:
#        p2 = intersection[0]
#        return p1.distance(p2)
#    else:
#        raise Exception("Could not calculate projected distance - Normal too short")

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        print 'Processing %s' % (fne)
        with open(fn, 'r') as f:
            int_sit = pickle.load(f)
        osm = transform_osm_to_cartesian(get_osm_data(int_sit))
        int_sit["track"] = transform_track_to_cartesian(int_sit["track"])
        entry_way = get_element_by_id(osm, int_sit["entry_way"])
        exit_way = get_element_by_id(osm, int_sit["exit_way"])
        entry_line, exit_line = get_way_lines(int_sit, osm)
        curve_secant = get_curve_secant_line(entry_line, exit_line)
        track_line = get_track_line(int_sit["track"])
        features = copy.deepcopy(_features)
        features["intersection_angle"] = get_intersection_angle(entry_line, exit_line)
        features["maxspeed_entry"] = get_maxspeed(entry_way)
        features["maxspeed_exit"] = get_maxspeed(exit_way)
        features["oneway_entry"] = get_oneway(entry_way)
        features["oneway_exit"] = get_oneway(exit_way)
        features["lane_distance_entry"] = get_lane_distance(entry_line, track_line, curve_secant)
        features["lane_distance_exit"] = get_lane_distance(exit_line, track_line, curve_secant)
        import json
        text = json.dumps(features, sort_keys=True, indent=4)
        print text
        
