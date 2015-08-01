#!/usr/bin/python
#coding:utf-8

import sys
import os.path
import random
from math import pi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GObject

GObject.threads_init()
Gdk.threads_init()

from gi.repository import OsmGpsMap as osmgpsmap
import gpxmanager
import osmmanager
import tracklayer

class TrackApp(object):
    def __init__(self):
        # All handlers for the signals defined in Glade
        self.handlers = {
            "onWinDelete": Gtk.main_quit,
            "onQuitButtonActivated": Gtk.main_quit,
            "onOSMButtonClicked": lambda b: self.toggle_mapsource(b,osmgpsmap.MapSource_t.OPENSTREETMAP),
            "onGStrButtonClicked": lambda b: self.toggle_mapsource(b,osmgpsmap.MapSource_t.GOOGLE_STREET),
            "onGSatButtonClicked": lambda b: self.toggle_mapsource(b,osmgpsmap.MapSource_t.GOOGLE_SATELLITE),
            "onOpenGPXClicked": self.on_open_gpx_clicked,
            "onTracklengthChanged": lambda w: self.set_track_moving_window(),
            "onOSMLoadClicked": self.on_osm_load_clicked,
            "updateLayerVisibility": self.update_layer_visibility
        }
        self.builder = Gtk.Builder()
        self.builder.add_from_file("trackwindow.glade")
        self.builder.connect_signals(self.handlers)
        self.win = self.builder.get_object("window1")
        self.map_box = self.builder.get_object("map_box")
        
        self.gpx_manager = gpxmanager.GPXManager()
        self.osm_manager = osmmanager.OSMManager()
        
        # Load icons for map buttons
        rb_osm = self.builder.get_object("radiobutton_osm")   
        rb_osm.set_image(Gtk.Image.new_from_file("data/media/osm_logo.png"))    
        rb_gstr = self.builder.get_object("radiobutton_gstreet")
        rb_gstr.set_image(Gtk.Image.new_from_file("data/media/gstreet_logo.png")) 
        rb_gsat = self.builder.get_object("radiobutton_gsatellite")
        rb_gsat.set_image(Gtk.Image.new_from_file("data/media/gsatellite_logo.png")) 
        
        self.label_info = self.builder.get_object("label_info")
        self.checkbutton_tracklength = self.builder.get_object("checkbutton_tracklength")
        self.adjustment_width = self.builder.get_object("adjustment_width")
        self.adjustment_center = self.builder.get_object("adjustment_center")
        self.adjustment_map_transparency = self.builder.get_object("adjustment_map_transparency")
        self.adjustment_osm_transparency = self.builder.get_object("adjustment_osm_transparency")
        self.adjustment_track_transparency = self.builder.get_object("adjustment_track_transparency")
        self.checkbutton_map_visible = self.builder.get_object("checkbutton_map_visible")
        self.checkbutton_osm_visible = self.builder.get_object("checkbutton_osm_visible")
        self.checkbutton_track_visible = self.builder.get_object("checkbutton_track_visible")
        
        self.create_osm()

        self.win.show_all()
        Gtk.main()
        
    def create_osm(self, reconstruct=False, map_source=osmgpsmap.MapSource_t.OPENSTREETMAP):
        """Creates or reconstructs the Map and Layers with given Map Source and coordinates"""
        print "Creating Osm..."
        lat, lon, zoom = (49.8725, 8.6498, 13)
        if reconstruct:
            lat, lon, zoom = (self.osm.props.latitude, self.osm.props.longitude, self.osm.props.zoom)
            self.map_box.remove(self.osm)
        self.osm = osmgpsmap.Map(map_source=map_source)
        self.osm.set_center_and_zoom(lat,lon, zoom)
        self.osm.layer_add(
                osmgpsmap.MapOsd(
                    show_dpad=True,
                    show_zoom=True,
                    show_crosshair=True)
        )
        # Add the extra drawing layers
        self.track_layer = tracklayer.TrackLayer(self.osm, self.gpx_manager, self.osm_manager)
        self.update_layer_visibility(None)
        self.osm.layer_add(self.track_layer)
        self.osm.set_size_request(400,400)
        self.map_box.pack_start(self.osm, True, True, 0)
        self.osm.show()
        
    def toggle_mapsource(self,b, map_source):
        if b.get_active():
            self.create_osm(reconstruct=True,map_source=map_source)
            
    def on_open_gpx_clicked(self, w):
        """Prompts a file chooser dialog to select a GPX-file from and opens it"""
        dialog = Gtk.FileChooserDialog("GPX-Datei öffnen", self.win,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        filter_gpx = Gtk.FileFilter()
        filter_gpx.set_name("GPX-Dateien")
        filter_gpx.add_pattern("*.gpx")
        dialog.add_filter(filter_gpx)
        
        filter_any = Gtk.FileFilter()
        filter_any.set_name("Alle Dateien")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.gpx_manager.open_gpx(dialog.get_filename())

        dialog.destroy()
        self.update_information_label()
        self.center_view_on_track()

    def update_information_label(self):
        gpx_str = ""
        
        if self.gpx_manager.has_track():
            (_,short_fn) = os.path.split(self.gpx_manager.gpx_filename)
            gpx_str = """<b>Datei:</b> %s
<b>Punkte:</b> %d
<b>Angezeigte Punkte:</b> %d""" % (short_fn, self.gpx_manager.track_point_count, self.gpx_manager.track_point_count)
        else:
            gpx_str = "Keine GPX-Datei geladen"
            
        self.label_info.set_markup(gpx_str)
            
    def set_track_moving_window(self):
        window_activated = self.checkbutton_tracklength.get_active()
        if window_activated:
            width = self.adjustment_width.get_value()
            center_pos = self.adjustment_center.get_value()/100.0
            self.gpx_manager.set_track_moving_window(width, center_pos)
        else:
            self.gpx_manager.set_track_moving_window(None, None)
        self.osm.map_redraw()
    
    def center_view_on_track(self):
        lon, lat = self.gpx_manager.get_track_window_iter().next()
        self.osm.set_center(lat, lon)
       
    def on_osm_load_clicked(self, w):
        """On maximum zoom level -> download osm data and store it"""
        zoom = self.osm.get_property("zoom")
        p1, p2 = self.osm.get_bbox()
        lat1, lon1 = p1.get_degrees()
        lat2, lon2 = p2.get_degrees()
        print lon1, lat1
        print lon2, lat2
        if (zoom >= 17):
            self.osm_manager.download_osm_map(lon1, lat1, lon2, lat2)
            self.osm.map_redraw()
        else:
            print "Zum Download der OSM-Daten bitte maximale Zoom-Stufe wählen"

    def update_layer_visibility(self, w):
        map_v = self.checkbutton_map_visible.get_active()
        osm_v = self.checkbutton_osm_visible.get_active()
        track_v = self.checkbutton_track_visible.get_active()
        map_t = self.adjustment_map_transparency.get_value()/100.0
        osm_t = self.adjustment_osm_transparency.get_value()/100.0
        track_t = self.adjustment_track_transparency.get_value()/100.0
        self.track_layer.set_layer_visibility(map_v, map_t, osm_v, osm_t, track_v, track_t)
        self.osm.map_redraw()

if __name__ == "__main__":
    TrackApp()
