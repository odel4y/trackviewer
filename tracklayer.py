#!/usr/bin/python
#coding:utf-8
from gi.repository import OsmGpsMap as osmgpsmap
from gi.repository import Gtk, Gdk
from gi.repository import GObject
from gi.repository import GdkPixbuf
import cairo

class BufferedLayer(GObject.GObject, osmgpsmap.MapLayer):
    """The BufferedLayer only rerenders when the Map dragging stops. When the map
    is moved only a drawing buffer is painted and moved accordingly (performance)"""
    def __init__(self, gpsmap):
        super(BufferedLayer, self).__init__()
        self.surface = None
        self._dragging = False
        self._drag_startx = self._drag_starty = 0
        self._drag_dx = self._drag_dy = 0
        self._clicked_handler = None
        gpsmap.connect("size-allocate", self.do_resize)
        gpsmap.connect("button-release-event", self.do_button_release)
        gpsmap.connect("motion-notify-event", self.do_motion_notify)
        
    def do_resize(self, gpsmap, allocation):
        map_alloc = gpsmap.get_allocation()
        print map_alloc.width, map_alloc.height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, map_alloc.width, map_alloc.height)
        gpsmap.map_redraw() # Force redraw when map is resized

    def do_draw(self, gpsmap, cr):
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
            if self._clicked_handler and abs(self._drag_dx)<2.0 and abs(self._drag_dy)<2.0:
                #Detecting a click on the map (maybe some track clicked)
                self._clicked_handler(gdkeventbutton.x, gdkeventbutton.y)
            self._drag_dx = self._drag_dy = 0
        return False
        
    def do_motion_notify(self, gpsmap, gdkeventmotion):
        if self._dragging:
            self._drag_dx = gdkeventmotion.x - self._drag_startx
            self._drag_dy = gdkeventmotion.y - self._drag_starty

    def set_clicked_handler(self, fun):
        self._clicked_handler = fun
        
GObject.type_register(BufferedLayer)

class TrackLayer(BufferedLayer):
    def __init__(self, gpsmap, gpxm, osmm):
        super(TrackLayer, self).__init__( gpsmap)
        self.gpx_manager = gpxm
        self.osm_manager = osmm
        self.map_visible = True
        self.osm_visible = True
        self.track_visible = True
        self.map_transparency = 1.0
        self.osm_transparency = 1.0
        self.track_transparency = 1.0

    def do_render(self, gpsmap):
        if self.surface:
            cr = cairo.Context(self.surface)
            cr.set_operator(cairo.OPERATOR_SOURCE)
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            cr.set_source_rgba (1, 1, 1, 1.0 - self.map_transparency)
            cr.paint()
            cr.set_operator(cairo.OPERATOR_OVER)
            if self.osm_manager.has_data() and self.osm_visible:
                cr.set_source_rgba (0, 0, 1, self.osm_transparency)
                cr.set_line_width(3.0)
                for way in self.osm_manager.get_way_iter():
                    if way["id"] == self.osm_manager.selected_way:
                        cr.set_source_rgba (1, 0.5, 0, self.osm_transparency)
                    else:
                        cr.set_source_rgba (0, 0, 1, self.osm_transparency)
                    init = True
                    for node in self.osm_manager.get_node_iter(way):
                        lon, lat = node["lon"], node["lat"]
                        osm_p = osmgpsmap.MapPoint.new_degrees(lat, lon)
                        (next_x, next_y) = gpsmap.convert_geographic_to_screen(osm_p)
                        if init:
                            cr.move_to(next_x, next_y)
                            init = False
                        else:
                            cr.line_to(next_x, next_y)
                    cr.stroke()
                if self.osm_manager.selected_node != None:
                    cr.set_source_rgba (1, 0, 1, self.osm_transparency)
                    cr.set_line_width(12.0)
                    node = self.osm_manager.get_node(self.osm_manager.selected_node)
                    lon, lat = node["lon"], node["lat"]
                    osm_p = osmgpsmap.MapPoint.new_degrees(lat, lon)
                    (next_x, next_y) = gpsmap.convert_geographic_to_screen(osm_p)
                    cr.move_to(next_x, next_y)
                    cr.line_to(next_x, next_y)
                    cr.stroke()
                    
            if self.gpx_manager.has_track() and self.track_visible:
                cr.set_source_rgba (1, 0, 0, self.track_transparency)
                cr.set_line_width(3.0)
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
            
    def set_layer_visibility(self, map_v, map_t, osm_v, osm_t, track_v, track_t):
        """Set the visibility and transparency of the respective data layers"""
        self.map_transparency = map_t
        self.osm_transparency = osm_t
        self.track_transparency = track_t
        if not map_v: self.map_transparency = 0.0 # We cannot stop the map component from drawing but we can hide it
        self.osm_visible = osm_v
        self.track_visible = track_v
        
GObject.type_register(TrackLayer)
