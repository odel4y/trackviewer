#!/usr/bin/python
#coding:utf-8

class IntersectionSituation:
    def __init__(self):
        self.entry_way = None           # way id
        self.entry_way_node = None      # node id
        self.exit_way = None            # way id
        self.exit_way_node = None       # node id
        self.intersection_node = None   # node id
        self.track = None               # list of (lon, lat, time) values

