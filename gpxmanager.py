import gpxpy
from gi.repository import OsmGpsMap as osmgpsmap
from math import floor
from osmapi import OsmApi

class GPXManager:
    def __init__(self):
        self._gpx_track = None
        self.gpx_filename = ""
        self.track_point_count = 0
        self._w_width = None
        self._w_center = None
        
    def open_gpx(self,fn):
        """Opens a gpx file and stores it"""
        with open(fn, 'r') as gpx_file:
            self._gpx_track = gpxpy.parse(gpx_file)
            self.track_point_count = self.get_track_point_count()
            self.gpx_filename = fn
            print 'Opened', fn

    def remove_gpx(self):
        self._gps_track = None
        
    def get_track_point_count(self):
        count = 0
        for track in self._gpx_track.tracks:
            for segment in track.segments:
                for point in segment.points:
                    count += 1
        return count

    def has_track(self): return self._gpx_track != None

    def get_track_iter(self, w_width=None, w_center=None):
        """Returns an iterator over the points of the track. Allows a definition of a window by width and center percentage"""
        if self.has_track():
            if (w_width != None) and (w_center != None):
                start_p, end_p = self._calc_borders_from_moving_window(w_width, w_center)
            else:
                start_p, end_p = 0, self.track_point_count
            for track in self._gpx_track.tracks:
                for segment in track.segments:
                    for i, point in enumerate(segment.points):
                        if start_p <= i < end_p:
                            yield (point.longitude, point.latitude)

    def get_track_window_iter(self):
        return self.get_track_iter(self._w_width, self._w_center)

    def _calc_borders_from_moving_window(self, width, center):
        """Restricts the track to a window of given width and a center position (center in percent)"""
        width = min(width, self.track_point_count)
        half_width = floor(width/2.0)
        center_p = int(round((self.track_point_count) * center))
        #if position is smaller than half the window length at the borders disregard it
        center_p = max(center_p, half_width)
        center_p = min(center_p, self.track_point_count - half_width)
        start_p = center_p - half_width
        end_p = center_p + half_width
        return (start_p, end_p)
        
    def has_osm_data(self): pass
        
    def download_osm_map(self, lon1, lat1, lon2, lat2):
        MyApi = OsmApi()
        print lon1, lat1
        print lon2, lat2
        #print MyApi.Map(min(lon1,lon2), min(lat1,lat2), max(lon1,lon2), max(lat1,lat2))
            
    def set_track_moving_window(self, w_width, w_center):
        self._w_width = w_width
        self._w_center = w_center
        
