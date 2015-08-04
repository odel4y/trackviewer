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
    
def get_element_by_id(osm, el_id):
    """Returns the """
    osm_iter = iter(osm)
    this_el = osm_iter.next()
    while this_el["id"] != el_id:
        try:
            this_el = osm_iter.next()
        except StopIteration:
            this_el = None
    return this_el
    
def get_line_string_from_node_ids(osm, nodes):
    """Construct a LineString from a list of Node IDs"""
    coords = []
    for node_id in nodes:
        this_node = get_element_by_id(osm, node_id)
        coords.append((this_node["x"], this_node["y"]))
    return shapely.LineString(coords)
    
def get_way_line_strings(int_sit, osm):
    """Return LineStrings for the actual entry and exit way separated by the intersection node both facing away from the intersection"""
    entry_way = get_element_by_id(osm, int_sit["entry_way"])
    entry_way_node_ids = entry_way["nodes"]
    if int_sit["entry_way"] == int_sit["exit_way"]: # One way for entry and exit -> split at intersection node
        cut_index = entry_way_node_ids.index(int_sit["intersection_node"])
        if entry_way_node_ids.index(int_sit["entry_way"]) < cut_index:
            entry_way_line_string = get_line_string_from_node_ids(osm, reversed(entry_way_node_ids[:cut_index+1]))
            exit_way_line_string = get_line_string_from_node_ids(osm, entry_way_node_ids[cut_index:])
        else:
            entry_way_line_string = get_line_string_from_node_ids(osm, entry_way_node_ids[cut_index:])
            exit_way_line_string = get_line_string_from_node_ids(osm, reversed(entry_way_node_ids[:cut_index+1]))
    else:
        exit_way = get_element_by_id(osm, int_sit["exit_way"])
        entry_way_node_ids = entry_way["nodes"]
        exit_way_node_ids = exit_way["nodes"]
        if int_sit["intersection_node"] == entry_way_node_ids[0]:
            entry_way_line_string = get_line_string_from_node_ids(osm, entry_way_node_ids)
        else:
            entry_way_line_string = get_line_string_from_node_ids(osm, reversed(entry_way_node_ids))
        if int_sit["intersection_node"] == exit_way_node_ids[0]:
            exit_way_line_string = get_line_string_from_node_ids(osm, exit_way_node_ids)
        else:
            exit_way_line_string = get_line_string_from_node_ids(osm, reversed(exit_way_node_ids))
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
        
        
