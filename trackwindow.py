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
            "onDeleteGPXClicked": self.on_delete_gpx_clicked,
            "onTreeViewCursorChanged": self.on_treeview_cursor_changed
        }
        self.builder = Gtk.Builder()
        self.builder.add_from_file("trackwindow.glade")
        self.builder.connect_signals(self.handlers)
        self.win = self.builder.get_object("window1")
        self.map_box = self.builder.get_object("map_box")
        self.create_osm()
        
        # Load icons for map buttons
        rb_osm = self.builder.get_object("radiobutton_osm")   
        rb_osm.set_image(Gtk.Image.new_from_file("data/media/osm_logo.png"))    
        rb_gstr = self.builder.get_object("radiobutton_gstreet")
        rb_gstr.set_image(Gtk.Image.new_from_file("data/media/gstreet_logo.png")) 
        rb_gsat = self.builder.get_object("radiobutton_gsatellite")
        rb_gsat.set_image(Gtk.Image.new_from_file("data/media/gsatellite_logo.png")) 
        
        self.treeview_gpx = self.builder.get_object("treeview_gpx")
        self.liststore_gpx = self.builder.get_object("liststore_gpx")
        
        self.gpx_manager = gpxmanager.GPXManager()
        
        self.win.show_all()
        Gtk.main()
        
    def create_osm(self, reconstruct=False, map_source=osmgpsmap.MapSource_t.OPENSTREETMAP):
        """Creates or reconstructs the Map with given Map Source and coordinates"""
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
        self.update_gpx_list()
        
    def on_delete_gpx_clicked(self, w):
        index = self.get_treeview_index()
        if index is not None:
            self.gpx_manager.remove_gpx()
            self.update_gpx_list()
            
    def on_treeview_cursor_changed(self, w):
        index = self.get_treeview_index()
        self.gpx_manager.set_selected(index)
        for track in self.gpx_manager.gpx_tracks:
            self.osm.track_add(track.get_osmgpsmap_track())
        
    def update_gpx_list(self):
        self.liststore_gpx.clear()
        for gpx_track in self.gpx_manager.gpx_tracks:
            (_,file_name) = os.path.split(gpx_track.get_file_name())
            self.liststore_gpx.append([file_name, gpx_track.get_point_count()])

    def get_treeview_index(self):
        selection = self.treeview_gpx.get_selection()
        selection.set_mode(Gtk.SelectionMode.BROWSE)
        model, iter = selection.get_selected()
        if iter is not None:
            path = self.liststore_gpx.get_path(iter)
            index = path.get_indices()[0]
            return index
        else:
            return None
if __name__ == "__main__":
    TrackApp()
