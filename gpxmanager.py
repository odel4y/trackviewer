import gpxpy
from gi.repository import OsmGpsMap as osmgpsmap

class GPXManager:
    def __init__(self):
        self.gpx_files = []
        
    def open_gpx(self,fn):
        """Opens a gpx file, stores it in interal list and returns its index"""
        self.gpx_files.append(GPXTrackContainer(fn))
        print 'Opened', fn
        return len(self.gpx_files)-1

    def remove_gpx(self, i):
        fn = self.gpx_files[i].get_file_name()
        try:
            self.gpx_files[i:i+1] = []
            print 'Removed', fn
        except:
            print 'Could not remove ', fn

class GPXTrackContainer:
    def __init__(self, fn):
        with open(fn, 'r') as gpx_file:
            self.gpx = gpxpy.parse(gpx_file)
        self.file_name = fn
        
    def get_file_name(self): return self.file_name
    
    def get_point_count(self):
        count = 0
        for track in self.gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    count += 1
        return count
    
    def get_osmgpsmap_track(self):
        track = osmgpsmap.Track()
