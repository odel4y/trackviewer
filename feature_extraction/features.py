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

def get_osm_data(int_sit):
    api = overpass.API()
    search_str = '(way(%d);way(%d););(._;>>;);out;' % \
                    (int_sit["entry_way"], int_sit["exit_way"])
    result = api.Get(search_str)
    return result["elements"]

def transform_to_cartesian(osm):
    """Transform the OSM longitude/latitude values to a cartesian system"""
    in_proj = pyproj.Proj(init='epsg:4326')    # Längen-/Breitengrad
    out_proj = pyproj.Proj(init='epsg:3857')   # Kartesische Koordinaten
    for el in osm:
        if "lon" in el:
            x1,y1 = el["lon"], el["lat"]
            x2,y2 = pyproj.transform(in_proj,out_proj,x1,y1)
            el["x"], el["y"] = x2, y2
    return osm
    
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
    
def get_way_line_strings(int_sit, osm):
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
    return np.arccos(np.dot(entry_v, exit_v)/(np.linalg.norm(entry_v) * np.linalg.norm(exit_v))) * normal

def get_maxspeed(osm, way):
    """Get the tagged maxspeed for a way. If no maxspeed is
    tagged try to guess it from the highway tag or adjacent ways"""
    if "maxspeed" in way["tags"]:
        return way["tags"]["maxspeed"]
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

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        print 'Processing', fne
        with open(fn, 'r') as f:
            int_sit = pickle.load(f)
        osm = transform_to_cartesian(get_osm_data(int_sit))
        entry_line, exit_line = get_way_line_strings(int_sit, osm)
        features = copy.deepcopy(_features)
        features["intersection_angle"] = get_intersection_angle(entry_line, exit_line)
        features["maxspeed_entry"] = get_maxspeed(osm, get_element_by_id(osm, int_sit["entry_way"]))
        features["maxspeed_exit"] = get_maxspeed(osm, get_element_by_id(osm, int_sit["exit_way"]))
        
        
