#!/usr/bin/python
#coding:utf-8
from gi.repository import OsmGpsMap as osmgpsmap
from gi.repository import Gtk, Gdk
from gi.repository import GObject
from gi.repository import GdkPixbuf
import cairo

class BufferedLayer(GObject.GObject, osmgpsmap.MapLayer):
    """The BufferedLayer only rerenders when the Map dragging stops. When the map
    is moved only a drawing buffer is painted and moved with the dragging (performance)"""
    def __init__(self, gpsmap):
        GObject.GObject.__init__(self)
        self.surface = None
        self._dragging = False
        self._drag_startx = self._drag_starty = 0
        self._drag_dx = self._drag_dy = 0
        gpsmap.connect("size-allocate", self.do_resize)
        gpsmap.connect("button-release-event", self.do_button_release)
        gpsmap.connect("motion-notify-event", self.do_motion_notify)
        
    def do_resize(self, gpsmap, allocation):
        map_alloc = gpsmap.get_allocation()
        print map_alloc.width, map_alloc.height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, map_alloc.width, map_alloc.height)
        gpsmap.map_redraw() # Force redraw when map is resized

    def do_draw(self, gpsmap, cr):
        if self.gpx_manager.has_track():
            cr.set_source_surface(self.surface, self._drag_dx, self._drag_dy)
            cr.paint()

    def do_render(self, gpsmap):
        pass

    def do_busy(self):
        return False

    def do_button_press(self, gpsmap, gdkeventbutton):
        if gdkeventbutton.button == 1: #Left mouse button
            self._dragging = True
            self._drag_startx = gdkeventbutton.x
            self._drag_starty = gdkeventbutton.y
        return False
        
    def do_button_release(self, gpsmap, gdkeventbutton):
        if gdkeventbutton.button == 1:
            self._dragging = False
            self._drag_dx = self._drag_dy = 0
        return False
        
    def do_motion_notify(self, gpsmap, gdkeventmotion):
        if self._dragging:
            self._drag_dx = gdkeventmotion.x - self._drag_startx
            self._drag_dy = gdkeventmotion.y - self._drag_starty
GObject.type_register(BufferedLayer)


class TrackLayer(BufferedLayer):
    def __init__(self, gpsmap, gpxm):
        BufferedLayer.__init__(self, gpsmap)
        self.gpx_manager = gpxm

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

GObject.type_register(TrackLayer)
