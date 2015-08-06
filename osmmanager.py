#!/usr/bin/python
#coding:utf-8
import overpass
from collections import namedtuple
from math import sqrt

class OSMManager:
    def __init__(self):
        self._osm = None
        self.selected_way = None
        self.selected_node = None

    def has_data(self): return self._osm != None

    def download_osm_map(self, lon1, lat1, lon2, lat2):
        api = overpass.API()
        search_str = 'way["highway"~"secondary|tertiary|residential|primary|primary_link|living_street|service|unclassified"](%.10f,%.10f,%.10f,%.10f);(._;>>;);out body;' % (min(lat1,lat2), min(lon1,lon2), max(lat1,lat2), max(lon1,lon2))
        result = api.Get(search_str)
        self._osm = result["elements"]
        #import json
        #text = json.dumps(result, sort_keys=True, indent=4)
        #with open('overpass-example.txt','w') as tf:
        #    tf.write(text)

    def get_way_iter(self):
        ways = [w for w in self._osm if w["type"]=="way"]
        for way in ways:
            #print "yield",way
            yield way

    def get_node_iter(self, way=None):
        if way != None:
            for n_id in way["nodes"]:
                n_dic = [n for n in self._osm if n["id"]==n_id][0]
                yield n_dic
        else:
            for el in self._osm:
                if el["type"] == "node":
                    yield el

    def get_node(self, node_id):
        for el in self._osm:
            if el["id"] == node_id:
                return el

    def projected_distance(self, x1, y1, x2, y2, xp, yp):
        """Calculate parallel distance of a point to a line (vector)"""
        try:
            a = (y2-y1)/(x2-x1)     # Can be zero if some nodes in OSM are duplicates
            b = -1.0
            c = y1 + (y1-y2)/(x2-x1)*x1
            dist = abs(a*xp + b*yp + c)/sqrt(a**2+b**2)
        except ZeroDivisionError:
            print 'Warning: Two nodes in OSM have the same coordinates'
            dist = 100000.0         # Return overtly large value
        return dist

    def distance(self, x1, y1, xp, yp):
        return sqrt((x1-xp)**2 + (y1-yp)**2)

    def get_closest_way_to_point(self, p_lon, p_lat):
        """Return the way that has the least distance to the given point"""
        min_way_dist = 100.0
        min_node_dist = 100.0
        min_way_id = 0
        min_node_id = 0
        for way in self.get_way_iter():
            last_lon, last_lat = None, None
            for node in self.get_node_iter(way):
                lon, lat = node["lon"], node["lat"]
                if last_lon != None and last_lat != None:
                    way_dist = self.projected_distance(lon, lat, last_lon, last_lat, p_lon, p_lat)
                    node_dist = self.distance(lon, lat, p_lon, p_lat)
                    # point within bounding box of this line?
                    extra_border = 0.0001
                    lon_within = min(lon,last_lon) - extra_border <= p_lon <= max(lon,last_lon) + extra_border
                    lat_within = min(lat,last_lat) - extra_border <= p_lat <= max(lat,last_lat) + extra_border
                    if way_dist < min_way_dist and lon_within and lat_within:
                        min_way_dist = way_dist
                        min_way_id = way["id"]
                    if node_dist < min_node_dist:
                        min_node_dist = node_dist
                        min_node_id = node["id"]
                last_lon, last_lat = lon, lat
        return min_way_id, min_node_id
