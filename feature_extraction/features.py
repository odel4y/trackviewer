#!/usr/bin/python
#coding:utf-8
import sys
import os.path
import pickle
import overpass
import pyproj
import shapely

# Features
# - Straßenwinkel zueinander
# - Geschwindigkeitsbegrenzung am Eingang ausgang
# - Fahrstreifenwahl am Eingang/Ausgang
#   + Angegeben durch Abstand der Fahrstreifenmitte von Mittellinie der Straße in Openstreetmap
# - Ist Eingang oder Ausgang Einbahnstraße?
features = {
    "street_angle": None,
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
    in_proj = pyproj.Proj(init='epsg:4326')    # Längen-/Breitengrad
    out_proj = pyproj.Proj(init='epsg:3857')   # Kartesische Koordinaten
    for el in osm:
        if "lon" in el:
            x1,y1 = el["lon"], el["lat"]
            x2,y2 = pyproj.transform(in_proj,out_proj,x1,y1)
            el["x"], el["y"] = x2, y2
    return osm
    
def get_line_string_from_node_ids(nodes):
    """Construct a LineString from a list of Node IDs"""
    coords = []
    for node_id in nodes:
        osm_iter = iter(osm)
        this_node = osm_iter.next()
        while this_node["id"] != node_id:
            this_node = osm_iter.next()
        coords.append((this_node["x"], this_node["y"]))
    return shapely.LineString(coords)
    
def get_way_line_strings(int_sit, osm):
    """Return LineStrings for the actual entry and exit way separated by the intersection node both facing away from the intersection"""
    if int_sit["entry_way"] == int_sit["exit_way"]: # One way for entry and exit -> split at intersection node
        osm_iter = iter(osm)
        this_way = osm_iter.next()
        while this_way["id"] != int_sit["entry_way"]:
            this_way = osm_iter.next()
        node_ids = this_way["nodes"]
        cut_index = node_ids.index(int_sit["intersection_node"])
        if node_ids.index(int_sit["entry_way"]) < cut_index:
            entry_way_line_string = get_line_string_from_node_ids(reversed(node_ids[:cut_index+1]))
            exit_way_line_string = get_line_string_from_node_ids(node_ids[cut_index:])
        else:
            entry_way_line_string = get_line_string_from_node_ids(node[cut_index:])
            exit_way_line_string = get_line_string_from_node_ids(reversed(node_ids[:cut_index+1]))
        return (entry_way_line_string, exit_way_line_string)
    
def get_street_angle():
    pass

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        fn = os.path.abspath(fn)
        fp, fne = os.path.split(fn)
        print 'Processing', fne
        with open(fn, 'r') as f:
            int_sit = pickle.load(f)
        osm = transform_to_cartesian(get_osm_data(int_sit))
        
        
