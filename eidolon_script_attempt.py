# Use this by first loading the 4 nifti files then running the script through the File menu

from eidolon import *
mgr:SceneManager=getSceneMgr()  # not needed but sorts out IDE variable resolution

import sys
import numpy as np
	
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


def _stupid_import():
	import matplotlib.pyplot as plt  # needs to be imported first in the main thread

result=mgr.callThreadSafe(_stupid_import)


import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class MplCanvas(FigureCanvasQTAgg):
	def __init__(self, parent=None, width=5, height=4, dpi=100,is_3d=False):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		super().__init__(self.fig)  # must come before axes are created
		
		self.axes=self.fig.gca(projection=("3d" if is_3d else None))
		
		self.setParent(parent)
		self.setFocusPolicy(Qt.StrongFocus)  # ensures key press events are passed correctly through matplotlib
		self.fig.tight_layout()
		
		
class MplNavWidget(QtWidgets.QWidget):
	def __init__(self, dpi=100,is_3d=False,parent=None):
		super().__init__(parent=parent)

		self.mpl_canvas = MplCanvas(self, dpi=dpi,is_3d=is_3d)
		self.fig=self.mpl_canvas.fig
		self.axes=self.mpl_canvas.axes

		self.mpl_toolbar = NavigationToolbar(self.mpl_canvas, self)

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0,0,0,0)
		layout.addWidget(self.mpl_toolbar)
		layout.addWidget(self.mpl_canvas)       
			
def _show_3d_plot():  # run by the above class when enter pressed
  
	# Here you would do the last of your processing and then show the 3d scatter plot of the landmarks 
	# and we'd save the results to a new scene object and/or to an output file 
  
	widg=MplNavWidget(is_3d=True)
	
	# temporary mesh plot to demonstrate 3d rendering works through matplotlib
	x=np.linspace(-6,6,30) # X coordinates
	y=np.linspace(-6,6,30) # Y coordinates
	X,Y=np.meshgrid(x,y) # Forming MeshGrid
	Z=np.sin(np.sqrt(X**2+Y**2))
	widg.axes.plot_surface(X,Y,Z) # plots the 3D surface plot
	
	mgr.win.createDock("MPL Widget",widg)
	return widg


def _select_images(dockparent, valmap):
	dockparent.close()
	
	# Traverse the selected images in the first panel, this code just prints out the name and the file it was loaded from
	# Instead you would want to pick out the four images, make sure they were chosen, load the data from their respective
	# nifti files again then run your code
	for k,v in valmap.items():
		obj=mgr.findObject(v)
		filenames=None
		
		if obj is not None:
			filenames=obj.plugin.getObjFiles(obj)
			
		print(k,v,filenames)
		
	# next step is to choose the points you want to remove by selecting them in the next view
		
	# create the next dock which will have the point-choosing scatter plot
	
	

def _show_panel():
	# need to run all code in a function which will be executed by the main thread, needed for Qt
	
	# mgr.objs is the list of loaded scene objects, pick out only the names of images
	names=[obj.getName() for obj in mgr.objs if isinstance(obj,ImageSceneObject)]
	
	# create named parameters used by an interface generator to create the panel with the four drop-down menus of image names
	params=[
		ParamDef("SAX","Short Axis",ParamType._strlist,names),
		ParamDef("LAX1","Long Axis 1",ParamType._strlist,names),
		ParamDef("LAX2","Long Axis 2",ParamType._strlist,names),
		ParamDef("LAX3","Long Axis 3",ParamType._strlist,names)
	]
	
	p=ParamPanel(params)  # create the widget with components made from the ParamDef objects above
	p.layout.setContentsMargins(5,5,5,5)
	
	button=QtWidgets.QPushButton("OK")
	p.layout.addWidget(button)
	
	mgr.win.createDock("Select Short and Long Axes Views",p)
	
	dockparent=mgr.win.dockWidgets[-1].parent()
	
	# when the button is clicked the next function in the chain is called passing in the needed parameters
	# button.clicked.connect(lambda:_select_images(dockparent,p.getParamMap()))
	from masks2ContoursScripts import masks2ContoursMain
	dockparent.close()
	widg = MplNavWidget()
	PyQt_objs = (mgr, widg)
	masks2ContoursMain.main(PyQt_objs)


	return p
	

result=mgr.callThreadSafe(_show_panel)  # runs _show_panel in the main thread