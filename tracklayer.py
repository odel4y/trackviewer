#!/usr/bin/python
#coding:utf-8
from gi.repository import OsmGpsMap as osmgpsmap
from gi.repository import Gtk, Gdk
from gi.repository import GObject
from gi.repository import GdkPixbuf
import cairo

def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider

class TrackLayer(GObject.GObject, osmgpsmap.MapLayer):
    def __init__(self, gpsmap, gpxm):
        GObject.GObject.__init__(self)
        #self.osm = gpsmap
        self.gpx_manager = gpxm
        self.surface = None
        gpsmap.connect("size-allocate", self.do_resize)
        
    def do_resize(self, gpsmap, allocation):
        map_alloc = gpsmap.get_allocation()
        print map_alloc.width, map_alloc.height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, map_alloc.width, map_alloc.height)

    def do_draw(self, gpsmap, cr):
        if self.gpx_manager.has_track():
            cr.set_source_surface(self.surface, 0, 0)
            cr.paint()
        print 'do_draw'

    def do_render(self, gpsmap):
        if self.gpx_manager.has_track():
            cr = cairo.Context(self.surface)
            cr.set_operator(cairo.OPERATOR_SOURCE)
            cr.set_source_rgba (0, 0, 1, 0.00)
            cr.paint()
            cr.set_operator(cairo.OPERATOR_OVER)
            cr.set_source_rgba (0, 0, 1, 1.00)
            init = True
            for lon, lat in self.gpx_manager.get_track_window_iter():
                osm_p = osmgpsmap.MapPoint.new_degrees(lat, lon)
                (next_x, next_y) = gpsmap.convert_geographic_to_screen(osm_p)
                if init:
                    cr.move_to(next_x, next_y)
                    init = False
                else:
                    cr.line_to(next_x, next_y)
            cr.stroke()

        print 'do_render'

    def do_busy(self):
        print 'do_busy'
        return False

    def do_button_press(self, gpsmap, gdkeventbutton):
        print 'do_button_press'
        return False
        
GObject.type_register(TrackLayer)
