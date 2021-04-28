from eidolon import *
from masks2ContoursScripts import masks2ContoursMain
mgr:SceneManager = getSceneMgr() # Not needed but sorts out IDE variable resolution

import sys
import os.path
import numpy as np
	
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

def _stupid_import():
	import matplotlib.pyplot as plt 

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

def main():
	# mgr.callThreadSafe(fn) runs the function fn in the main thread.
	mgr.callThreadSafe(_stupid_import)
	result = mgr.callThreadSafe(_show_panel) 
 
class MplCanvas(FigureCanvasQTAgg):
	def __init__(self, parent = None, width = 5, height = 4, dpi = 100, is_3d=False):
		self.fig = Figure(figsize = (width, height), dpi = dpi)
		super().__init__(self.fig)  # must come before axes are created
		
		self.axes = self.fig.gca(projection = ("3d" if is_3d else None))
		
		self.setParent(parent)
		self.setFocusPolicy(Qt.StrongFocus)  # ensures key press events are passed correctly through matplotlib
		self.fig.tight_layout()
		
class MplNavWidget(QtWidgets.QWidget):
	def __init__(self, dpi = 100, is_3d = False, parent=None):
		super().__init__(parent = parent)

		self.mpl_canvas = MplCanvas(self, dpi = dpi,is_3d = is_3d)
		self.fig = self.mpl_canvas.fig
		self.axes = self.mpl_canvas.axes

		self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self.mpl_toolbar)
		layout.addWidget(self.mpl_canvas)

MENU_KEYS = ["SA", "SA_name", "LA1", "LA2", "LA3", "LA1_name", "LA2_name", "LA3_name"]
		
def _select_images(dockparent, valmap):
	dockparent.close()
	
	# Load the filepaths for the NIFTI files.
	dict = {}
	for key, value in valmap.items():
		obj = mgr.findObject(value)
		if obj is not None:
			filenames = obj.plugin.getObjFiles(obj)
			dict[key] = filenames[0]
	
	# Check that all required files have been loaded.
	if not set(MENU_KEYS).issubset(set(dict.keys())):
		print("Error!")
		# TO-DO: actually handle the error.

	# Prepare parameters to pass to masks2Contours.
	imgName = dict["SA_name"]
	segName = dict["SA"]
	LA_names = [dict["LA1_name"], dict["LA2_name"], dict["LA3_name"]]
	LA_segs = [dict["LA1"], dict["LA2"], dict["LA3"]]

	# Get the folder in which these files reside, if all files do indeed live in the same folder.
	imgFldr = os.path.dirname(imgName)
	segFldr = os.path.dirname(segName)
	LA_names_fldrs = [os.path.dirname(x) for x in LA_names]
	LA_segs_fldrs = [os.path.dirname(x) for x in LA_segs]
	fldrSet = {imgFldr, segFldr}.union(LA_names_fldrs).union(LA_segs_fldrs)
	if len(fldrSet) == 1: # if all folders containing the files represented by imgName, segName, LA_names, LA_segs are the same
		fldr = imgFldr
	else:
		print("The image files and segmentations should all be in the same folder", file = sys.stderr)

	# The filepath returned by os.path.dirname() doesn't end with a backslash, as it should, so we'll add one.
	fldr += "\\"
	
	# Run the masks2Contours script, passing along fldr and (mgr, widg).
	widg = MplNavWidget()
	PyQt_objs = (mgr, widg)
	masks2ContoursMain.main(fldr, imgName, segName, LA_names, LA_segs, PyQt_objs)
	
def _show_panel():
	# mgr.objs is the list of loaded scene objects.
	names = [obj.getName() for obj in mgr.objs if isinstance(obj, ImageSceneObject)]

	# Set up named parameters used by an interface generator.
	params = [
		ParamDef(MENU_KEYS[0],"Short axis (segmentation)",ParamType._strlist,names),
		ParamDef(MENU_KEYS[1], "Short axis (header metadata)", ParamType._strlist, names),
		ParamDef(MENU_KEYS[2],"Long axis 2ch (segmentation)",ParamType._strlist,names),
		ParamDef(MENU_KEYS[3],"Long axis 3ch (segmentation)",ParamType._strlist,names),
		ParamDef(MENU_KEYS[4],"Long axis 4ch (segmentation)",ParamType._strlist,names),
		ParamDef(MENU_KEYS[5], "Long axis 2ch (header metadata)", ParamType._strlist, names),
		ParamDef(MENU_KEYS[6], "Long axis 3ch (header metadata)", ParamType._strlist, names),
		ParamDef(MENU_KEYS[7], "Long axis 4ch (header metadata)", ParamType._strlist, names)
	]
	
	# Use "params", the list of named parameters, to create the panel with the four drop-down menus of image names.
	p = ParamPanel(params)
	p.layout.setContentsMargins(5, 5, 5, 5)
	
	# Set up a button.
	button = QtWidgets.QPushButton("OK")
	p.layout.addWidget(button)

	# Set up the dock that will contain the dropdown menus.
	mgr.win.createDock("Select Short and Long Axes Views", p)
	dockparent = mgr.win.dockWidgets[-1].parent()
	
	# Register the button's callback function.
	button.clicked.connect(lambda:_select_images(dockparent, p.getParamMap()))
	
	return p

main()