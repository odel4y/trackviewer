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

class CustomLayer(GObject.GObject, osmgpsmap.MapLayer):
    def __init__(self):
        GObject.GObject.__init__(self)

    def do_draw(self, gpsmap, cr):
        (p1,_) = gpsmap.get_bbox()
        print gpsmap.get_property("longitude")
        cr.set_source_rgba (0, 0, 1, 0.40);
        cr.move_to(100,100)
        cr.line_to(150,150)
        cr.stroke()
        print 'do_draw'

    def do_render(self, gpsmap):
        print 'do_render'

    def do_busy(self):
        print 'do_busy'
        return False

    def do_button_press(self, gpsmap, gdkeventbutton):
        print 'do_button_press'
        return False
GObject.type_register(CustomLayer)
