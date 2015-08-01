#!/usr/bin/python
#coding:utf-8
import overpass
from collections import namedtuple

class OSMManager:
    def __init__(self):
        self._osm = None
    
    def has_data(self): return self._osm != None
    
    def download_osm_map(self, lon1, lat1, lon2, lat2):
        api = overpass.API()
        search_str = 'way["highway"~"secondary|tertiary|residential|primary|primary_link"](%.10f,%.10f,%.10f,%.10f);(._;>>;);out body;' % (min(lat1,lat2), min(lon1,lon2), max(lat1,lat2), max(lon1,lon2))
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
            
    def get_node_iter(self, way):
        for n_id in way["nodes"]:
            n_dic = [n for n in self._osm if n["id"]==n_id][0]
            yield n_dic
            
