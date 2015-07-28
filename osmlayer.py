#!/usr/bin/python
#coding:utf-8
from gi.repository import OsmGpsMap as osmgpsmap
from gi.repository import Gtk, Gdk, cairo
from gi.repository import GObject
from gi.repository import GdkPixbuf

def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider

class OSMLayer(GObject.GObject, osmgpsmap.MapLayer):
    def __init__(self, gpxm):
        GObject.GObject.__init__(self)
        self.gpx_manager = gpxm
        self.w_width = None
        self.w_position = None

    def do_draw(self, gpsmap, cr):
#        if self.gpx_manager.has_track():
#            cr.set_source_rgba (0, 0, 1, 1.00)
#            init = True
#            for lon, lat in self.gpx_manager.get_track_point_iter(self.w_width, self.w_position):
#                osm_p = osmgpsmap.MapPoint.new_degrees(lat, lon)
#                (next_x, next_y) = gpsmap.convert_geographic_to_screen(osm_p)
#                if init:
#                    cr.move_to(next_x, next_y)
#                    init = False
#                else:
#                    cr.line_to(next_x, next_y)
#            cr.stroke()

        print 'do_draw'

    def do_render(self, gpsmap):

        print 'do_render'

    def do_busy(self):
        print 'do_busy'
        return False

    def do_button_press(self, gpsmap, gdkeventbutton):
        print 'do_button_press'
        return False

    def download_osm_bbox(self, lon1, lat1, lon2, lat2):
        pass
        
GObject.type_register(OSMLayer)
