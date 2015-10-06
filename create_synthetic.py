#!/usr/bin/python
#coding:utf-8
# Create synthetic osm data and int_sit
from trackwindow import _intersection_situation
import copy
from extract_features import split_line, create_sample, upsample_line
import datetime

def create_intersection_data(line, split_dist, normalized=False, tags={}, explicit_features={}):
    """Accepts a LineString and a distance where the intersection center is and creates fake osm data for it."""
    node_ids = {}
    osm = []
    int_sit = copy.deepcopy(_intersection_situation)
    entry_line, exit_line = split_line(line, split_dist, normalized=normalized)

    # Assign ID to every coord in LineStrings
    for coord in entry_line.coords[:] + exit_line.coords[:]:
        if coord not in node_ids: node_ids[coord] = create_unique_id()

    # Make OSM representation of nodes
    int_sit["entry_way_node"] = node_ids[entry_line.coords[0]]
    int_sit["exit_way_node"] = node_ids[exit_line.coords[-1]]
    int_sit["intersection_node"] = node_ids[entry_line.coords[-1]]
    osm.extend( [get_node_from_coord(coord, coord_id) for coord, coord_id in node_ids.iteritems()] )

    # Make OSM representation of ways
    int_sit["entry_way"] = create_unique_id()
    int_sit["exit_way"] = create_unique_id()
    try:
        osm.append(get_way_from_line(entry_line, int_sit["entry_way"], node_ids, tags["entry_way"]))
    except KeyError:
        osm.append(get_way_from_line(entry_line, int_sit["entry_way"], node_ids))
    try:
        osm.append(get_way_from_line(exit_line, int_sit["exit_way"], node_ids, tags["exit_way"]))
    except KeyError:
        osm.append(get_way_from_line(exit_line, int_sit["exit_way"], node_ids))

    # Just use OSM path as track
    # Upsample to allow for extraction of vehicle speed
    track_line = upsample_line(line, 5)
    x,y = zip(*track_line.coords[:])
    times = [datetime.date.today() + datetime.timedelta(days=i) for i in range(len(x))]
    print times
    int_sit["track"] = zip(x,y,times)

    return int_sit, osm

def get_node_from_coord(coord, coord_id):
    node = {
        "id": coord_id,
        "x": coord[0],
        "y": coord[1],
        "tags": {},
        "type": "node"
    }
    return node

def get_way_from_line(line, way_id, node_ids, tags={}):
    way = {
        "id": way_id,
        "nodes": [node_ids[coord] for coord in line.coords[:]],
        "tags": tags,
        "type": "way"
    }
    return way

def create_unique_id():
    create_unique_id.counter += 1
    return create_unique_id.counter
create_unique_id.counter = 0
