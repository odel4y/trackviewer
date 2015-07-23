import gpxpy
from gi.repository import OsmGpsMap as osmgpsmap

class GPXManager:
    def __init__(self):
        self.gpx_tracks = []
        
    def open_gpx(self,fn):
        """Opens a gpx file, stores it in interal list and returns its index"""
        self.gpx_tracks.append(GPXTrackContainer(fn))
        print 'Opened', fn
        return len(self.gpx_tracks)-1

    def remove_gpx(self, i):
        fn = self.gpx_tracks[i].get_file_name()
        try:
            self.gpx_tracks[i:i+1] = []
            print 'Removed', fn
        except:
            print 'Could not remove ', fn
            
    def set_selected(self, i):
        for j, gpx_file in enumerate(self.gpx_tracks):
            if i == j:
                gpx_file.set_selected(True)
            else:
                gpx_file.set_selected(False)

class GPXTrackContainer:
    def __init__(self, fn):
        with open(fn, 'r') as gpx_file:
            self.gpx = gpxpy.parse(gpx_file)
        self.file_name = fn
        self.selected = False
        
    def get_file_name(self): return self.file_name
    
    def get_selected(self): return self.selected
    
    def set_selected(self, s): self.selected = s
    
    def get_point_count(self):
        count = 0
        for track in self.gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    count += 1
        return count
    
    def get_osmgpsmap_track(self):
        track = osmgpsmap.MapTrack()
        for t in self.gpx.tracks:
            for segment in t.segments:
                for point in segment.points:
                    track.add_point(osmgpsmap.MapPoint.new_degrees(point.latitude, point.longitude))
        return track
