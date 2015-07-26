import gpxpy
from gi.repository import OsmGpsMap as osmgpsmap

class GPXManager:
    def __init__(self):
        self.gpx_track = None
        self.gpx_filename = ""
        
    def open_gpx(self,fn):
        """Opens a gpx file and stores it"""
        with open(fn, 'r') as gpx_file:
            self.gpx_track = gpxpy.parse(gpx_file)
            self.gpx_filename = fn
            print 'Opened', fn

    def remove_gpx(self):
        self.gps_track = None
        
    def get_gpx_point_count(self):
        count = 0
        for track in self.gpx_track.tracks:
            for segment in track.segments:
                for point in segment.points:
                    count += 1
        return count

