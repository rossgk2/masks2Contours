import sys
sys.path.append("masks2ContoursScripts")

import csv
from types import SimpleNamespace

import masks2ContoursMainUtil as mut
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage import measure
from skimage import morphology

import MONAIutils
import masks2ContoursUtil as ut
from SelectFromCollection import SelectFromCollection

class InputsHolder:
	def __init__(self, dict, sliceIndex, SA_LA):
		self.SA_LA = str.lower(SA_LA)
		self.sliceIndex = sliceIndex

		if not (self.SA_LA == "sa" or self.SA_LA == "la"):
			raise ValueError("SA_LA must either be \"SA\" or \"LA\".")

		self.dict = {}
		for key in dict.keys():
			self.dict[key] = dict[key]

	def __getitem__(self, key):
		if self.SA_LA == "sa":
			return self.dict[key]
		else: #SA_LA == "la"
			print("__getitem__() for LA called. key = {}, sliceIndex = {}".format(key, self.sliceIndex))
			return self.dict[key][self.sliceIndex]

def masks2ContoursSA(segName, frameNum, config):
	# Read the 3D segmentation, transform matrix, and other geometry info from the NIFTI file.
	(seg, transform, pixSpacing) = readFromNIFTI(segName, frameNum)
	numSlices = seg.shape[2]

	# Separate out LV endo, LV epi, and RV endo segmentations.
	LVendo = seg == 1
	LVepi = (seg <= 2) * (seg > 0)
	RVendo = seg == 3

	# Initialize variables.
	ub = config.upperBdNumContourPts
	LVendoContours = np.zeros([ub, 3, numSlices])
	LVepiContours = np.zeros([ub, 3, numSlices])
	RVFWendoContours = np.zeros([ub, 3, numSlices])
	RVFWepiContours = np.zeros([ub, 3, numSlices])
	RVseptContours = np.zeros([ub, 3, numSlices])
	RVinserts = np.zeros([2, 3, numSlices])

	# Loop over slices and get contours one slice at a time.
	for i in range(0, numSlices):
		# Note that transform and pixSpacing are the same in every iteration.
		inputsHolder = InputsHolder(dict = {"LVendo" : LVendo[:, :, i], "LVepi" : LVepi[:, :, i], "RVendo" : RVendo[:, :, i],
										"transform": transform, "pixSpacing" : pixSpacing},
										sliceIndex = None, SA_LA = "SA")
		outputsHolder = SimpleNamespace(LVendoContours = LVendoContours, LVepiContours = LVepiContours,
							 RVseptContours = RVseptContours, RVFWendoContours = RVFWendoContours,
							 RVFWepiContours = RVFWepiContours, RVinserts = RVinserts)
		slice2ContoursPt1(inputsHolder = inputsHolder, outputsHolder = outputsHolder, config = config,
						  sliceIndex = i, numLAslices = None, SA_LA = "SA", PyQt_objs = None, mainObjs = None)

	# Now, calculate weights for RV insertion points.

	# Loop through anterior (1) and posterior (2) sides for fitting a line.
	RVinsertsWeights = np.zeros((2, numSlices)) # This will store errors.
	for i in range(0, 2):

		# In the MATLAB script, the following step essentially amounted to horizontally stacking _inserts1^T and _inserts2^T. (^T is matrix transpose).
		# Since transposing a 1 x n ndarray produces a 1 x n ndarray, rather than a n x 1 ndarray (this happens because numpy
		# tries to make matrix transposition more efficent, since tranposition is a very predictable shuffle of old data)
		# we do a different but equivalent operation here. That is, we take the transpose of the ndarray obtained by stacking
		# _inserts2 below _inserts1.
		_inserts1 = np.squeeze(RVinserts[i, :, :])
		_inserts2 = np.linspace(0, numSlices - 1, numSlices) #do we need to use 0, slices - 1 for first two args of linspace?
		inserts = np.vstack((_inserts1, _inserts2))
		inserts = inserts.transpose()
		inserts = inserts[np.all(inserts[:, 0:2], 1), :] # Get rid of rows with zeros in the first three columns.

		points = inserts[:, 0:3]
		indices = inserts[:, 3].astype(int)

		# Sort RV insert points by error, and save the normalized error in RVinsertsWeights.
		err = ut.fitLine3D(points)
		RVinsertsWeights[i, indices] = np.abs(err)/np.max(np.abs(err))

	# Return a 2-tuple of dictionaries.
	return ({"LVendo" : LVendoContours,
			"LVepi" : LVepiContours,
			"RVFWendo" : RVFWendoContours,
			"RVFWepi" : RVFWepiContours,
			"RVsept" : RVseptContours } ,
			{"RVinserts" : RVinserts, "RVinsertsWeights" : RVinsertsWeights})

def masks2ContoursLA(LA_segs, frameNum, numSlices, config, PyQt_objs, mainObjs):
	# Precompute (more accurately, "pre-read") endoLV, epiLV, endoRV for each slice.
	# Also precompute transform and pix_spacing for each slice.
	(LVendoList, LVepiList, RVendoList, transformList, pixSpacingList) = ([], [], [], [], [])
	for i in range(0, numSlices):
		(seg, transform, pixSpacing) = readFromNIFTI(LA_segs[i], frameNum)
		LVendoList.append(np.squeeze(seg == 1))
		LVepiList.append(np.squeeze((seg <= 2) * (seg > 0)))
		RVendoList.append(np.squeeze(seg == 3))
		transformList.append(transform)
		pixSpacingList.append(pixSpacing)

	# Initialize variables.
	ub = config.upperBdNumContourPts
	LVendoContours = np.zeros([ub, 3, numSlices])
	LVepiContours = np.zeros([ub, 3, numSlices])
	RVFWendoContours = np.zeros([ub, 3, numSlices])
	RVFWepiContours = np.zeros([ub, 3, numSlices])
	RVseptContours = np.zeros([ub, 3, numSlices])

	# Recurse over slices and get contours one slice at a time.
	inputsHolder = InputsHolder(dict = {"LVendo" : LVendoList, "LVepi" : LVepiList, "RVendo" : RVendoList,
										"transform": transformList, "pixSpacing" : pixSpacingList},
								sliceIndex = 0, SA_LA = "LA")
	outputsHolder = SimpleNamespace(LVendoContours = LVendoContours, LVepiContours = LVepiContours,
									RVseptContours = RVseptContours, RVFWendoContours = RVFWendoContours,
									RVFWepiContours = RVFWepiContours)
	slice2ContoursPt1(inputsHolder = inputsHolder, outputsHolder = outputsHolder, config = config, sliceIndex = 0,
					  numLAslices = numSlices, SA_LA = "LA", PyQt_objs = PyQt_objs, mainObjs = mainObjs)

def slice2ContoursPt1(inputsHolder, outputsHolder, config, sliceIndex, numLAslices, SA_LA, PyQt_objs, mainObjs):
	print("slice2ContoursPt1() called. sliceIndex = {}".format(sliceIndex))
	# Check validity of SA_LA.
	SA_LA = SA_LA.lower()
	if not(SA_LA == "sa" or SA_LA == "la"):
		raise ValueError("SA_LA must either be \"SA\" or \"LA\".")

	# Get contours for each slice in the different regions of the heart. In the following variable names,
	# "CS" stands for "contour slice".

	LVendoCS = getContoursFromMask(inputsHolder["LVendo"], irregMaxSize = 20)
	LVepiCS = getContoursFromMask(inputsHolder["LVepi"], irregMaxSize = 20)
	RVendoCS = getContoursFromMask(inputsHolder["RVendo"], irregMaxSize = 20)
	print("RVendoCS.shape = {}".format(RVendoCS.shape))

	# Swap the axes. We have to do this due to the way that the nibabel module handles NIFTI files.
	LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
	LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
	RVendoIsEmpty = RVendoCS is None or RVendoCS.size == 0

	if not LVendoIsEmpty:
		LVendoCS[:, [0, 1]] = LVendoCS[:, [1, 0]]
	if not LVepiIsEmpty:
		LVepiCS[:, [0, 1]] = LVepiCS[:, [1, 0]]
	if not RVendoIsEmpty:
		RVendoCS[:, [0, 1]] = RVendoCS[:, [1, 0]]

	# Differentiate contours for RV free wall and RV septum.
	[RVseptCS, ia, ib] = ut.sharedRows(LVepiCS, RVendoCS)
	RVFW_CS = ut.deleteHelper(RVendoCS, ib, axis = 0) # Delete the rows whose indices are in ib.

	# Remove RV septum points from the LV epi contour.
	LVepiCS = ut.deleteHelper(LVepiCS, ia, axis = 0)  # In LVepiCS, delete the rows with index ia.

	# If doing long axis, remove line segments at base which are common to LVendoCS and LVepiCS.
	if SA_LA == "la":
		[wont_use_this_var, ia, ib] = ut.sharedRows(LVendoCS, LVepiCS)
		LVendoCS = ut.deleteHelper(LVendoCS, ia, axis = 0)
		LVepiCS = ut.deleteHelper(LVepiCS, ib, axis = 0)

	# Since we've modified LVendoCS, LVepiCS, and RVFW_CS, we should re-check whether these variables are "empty".
	LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
	LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
	RVendoIsEmpty = RVFW_CS is None or RVFW_CS.size == 0

	# The rest of this function, broadly speaking, uses LVendoCS, LVepiCS, RVFW_CS as input for the function contoursToImageCoords(),
	# which writes to outputsHolder.LVendoContours, outputsHolder.LVepiContours, and other similarly-named variables.
	# (contoursToImageCoords() writes to its fourth argument as a side-effect).

	# Input: LVendoCS, output: outputsHolder.LVendoContours
	if not LVendoIsEmpty:
		LVendoCS = cleanContours(LVendoCS, config.downsample)
		contoursToImageCoords(LVendoCS, inputsHolder["transform"], sliceIndex, outputsHolder.LVendoContours, SA_LA)

	# Input: LVepiCS, output: outputsHolder.LVepiContours
	if not LVepiIsEmpty:
		LVepiCS = cleanContours(LVepiCS, config.downsample)
		contoursToImageCoords(LVepiCS, inputsHolder["transform"], sliceIndex, outputsHolder.LVepiContours, SA_LA)

	# Do some computations to set up for doing the same thing as in the above two commented blocks of code
	# for the RV. contoursToImageCoords() is used for the RV in slice2ContoursPt2(), which is called by the below code.
	print("slice index is {}".format(sliceIndex))
	tmpRV_insertIndices = None
	if not RVendoIsEmpty:
		print("inside the if statement coorresponding to \"# RV\"")
		RVFW_CS = cleanContours(RVFW_CS, config.downsample)
		RVseptCS = cleanContours(RVseptCS, config.downsample)

		# If doing short axis, get the indices corresponding to insertion points in the RV.
		if SA_LA == "sa":
			tmpRV_insertIndices = ut.getRVinsertIndices(RVFW_CS)

		# If doing long axis, remove basal line segment in RVFW LA contour.
		RVEndoIsEmptyAfterCleaning = (RVFW_CS is None or RVseptCS is None) or (RVFW_CS.size == 0 or RVseptCS.size == 0)
		if SA_LA == "la" and not RVEndoIsEmptyAfterCleaning:
			print("inside the if statement that does the lasso select. sliceIndex = {}".format(sliceIndex))

			# Unpack PyQt_objs.
			(mgr, widg) = PyQt_objs

			# Set up the scatter plot, "lassoPts", that will be passed to the lasso selector.
			# The argument "title" to subplotHelper cannot be accessed later, so it doesn't matter what it is.
			lassoPts = subplotHelper(widg.axes, title = None, maskSlice = np.squeeze(inputsHolder["RVendo"]), contours = RVFW_CS, color = "green", size = 20)

			# Create the lasso selector.
			pt2Data = SimpleNamespace(inputsHolder = inputsHolder, outputsHolder = outputsHolder, RVseptCS = RVseptCS, RVFW_CS = RVFW_CS, 
			tmpRV_insertIndices = tmpRV_insertIndices, config = config, SA_LA = SA_LA, PyQt_objs = PyQt_objs, mainObjs = mainObjs)
			passToLasso = SimpleNamespace(pt2Data = pt2Data, sliceIndex = sliceIndex, numLAslices = numLAslices)
			print("passToLasso.sliceIndex = {}".format(passToLasso.sliceIndex))
			lassoSelector = SelectFromCollection(passToLasso, ax = widg.axes, collection = lassoPts)
			print("The new lasso selector has been created")
			print("From within masks2Contours.py, lassoSelector.passToLasso.sliceIndex = {}".format(lassoSelector.passToLasso.sliceIndex))

			# This causes plot-with-interactive-lasso-selector to appear. When the user presses Enter,
			# the callback function lassoSelector.key_press() executes, and removes points from RVFW_CS
			# that were lasso-selected by the user. lassoSelector.key_press() then calls slice2ContoursPt2().
			widg.lasso = lassoSelector # Prevents lassoSelector from getting garbage collected.
			mgr.win.createDock("MPL Widget", widg)
			lassoSelector.parent_widg = mgr.win.dockWidgets[-1].parent() # dock widget contains `widg`, parent of that is dock itself
			print("The new lasso selector has been hooked up")
		else:
			print("inside the else case that executes when no lasso select is to be performed")
			# Wrap up stuff we've computed so that it can be passed to slice2ContoursPt2().
			pt2Data = SimpleNamespace(inputsHolder = inputsHolder, outputsHolder = outputsHolder, RVseptCS = RVseptCS, RVFW_CS = RVFW_CS, 
			tmpRV_insertIndices = tmpRV_insertIndices, config = config, SA_LA = SA_LA, PyQt_objs = PyQt_objs, mainObjs = mainObjs)

			slice2ContoursPt2(pt2Data = pt2Data, sliceIndex = sliceIndex, numLASlices = numLAslices)
	else:
		print("inside the else case that matches the \"if\" for \"#RV\"")
		# Wrap up stuff we've computed so that it can be passed to slice2ContoursPt2().
		pt2Data = SimpleNamespace(inputsHolder = inputsHolder, outputsHolder = outputsHolder, RVseptCS = RVseptCS, RVFW_CS = RVFW_CS, 
			tmpRV_insertIndices = tmpRV_insertIndices, config = config, SA_LA = SA_LA, PyQt_objs = PyQt_objs, mainObjs = mainObjs)

		slice2ContoursPt2(pt2Data = pt2Data, sliceIndex = sliceIndex, numLASlices = numLAslices)

def slice2ContoursPt2(pt2Data, sliceIndex, numLASlices):
	# Unpack pt2Data.
	transform = pt2Data.inputsHolder["transform"]
	LVendoContours = pt2Data.outputsHolder.LVendoContours
	LVepiContours = pt2Data.outputsHolder.LVepiContours
	RVseptContours = pt2Data.outputsHolder.RVseptContours
	RVFWendoContours = pt2Data.outputsHolder.RVFWendoContours
	RVFWepiContours = pt2Data.outputsHolder.RVFWepiContours
	RVseptCS = pt2Data.RVseptCS
	RVFW_CS = pt2Data.RVFW_CS
	config = pt2Data.config
	SA_LA = pt2Data.SA_LA

	if SA_LA == "sa":
		tmpRV_insertIndices = pt2Data.tmpRV_insertIndices
		RVinserts = pt2Data.outputsHolder.RVinserts

	# Write to RVseptContours and RVFWendoContours.
	contoursToImageCoords(RVseptCS, transform, sliceIndex, RVseptContours, SA_LA)
	contoursToImageCoords(RVFW_CS, transform, sliceIndex, RVFWendoContours, SA_LA)

	# If doing short axis, save RV insert coordinates.
	if SA_LA == "sa" and (tmpRV_insertIndices is not None and tmpRV_insertIndices.size != 0):
		RVinserts[:, :, sliceIndex] = RVFWendoContours[tmpRV_insertIndices[0:1 + 1], :, sliceIndex]

	# Calculate RV epicardial wall by applying a thickness to RV endocardium.
	if SA_LA == "sa":
		RVEndoNormals = ut.lineNormals2D(RVFW_CS)
		RVepiSC = RVFW_CS - np.ceil(config.rvWallThickness / pt2Data.inputsHolder["pixSpacing"]) * RVEndoNormals \
			if RVFW_CS.size != 0 and RVEndoNormals.size != 0 else np.array([])
		contoursToImageCoords(RVepiSC, transform, sliceIndex, RVFWepiContours, "SA")
	else: # SA_LA == "la"
		print("slice2ContoursPt2() called for LA")
		if sliceIndex + 1 < numLASlices:
			pt2Data.inputsHolder.sliceIndex += 1
			slice2ContoursPt1(inputsHolder = pt2Data.inputsHolder, outputsHolder = pt2Data.outputsHolder,
							  config = pt2Data.config, sliceIndex = sliceIndex + 1, numLAslices = numLASlices,
							  SA_LA = pt2Data.SA_LA, PyQt_objs = pt2Data.PyQt_objs, mainObjs = pt2Data.mainObjs)
		else:
			LAcontours = {"LVendo": LVendoContours, "LVepi": LVepiContours, "RVFWendo": RVFWendoContours,
						  "RVFWepi": RVFWepiContours, "RVsept": RVseptContours}
			mainObjs = pt2Data.mainObjs
			finishUp(SAresults = mainObjs.SAresults, LAcontours = LAcontours, frameNum = mainObjs.frameNum,
					 fldr = mainObjs.fldr, imgName = mainObjs.imgName)

# Helper function for converting contours to an image coordinate system. This function writes to "contours".
def contoursToImageCoords(maskSlice, transform, sliceIndex, contours, SA_LA):
	SA_LA = SA_LA.lower()
	if SA_LA == "sa":
		thirdComp = sliceIndex
	elif SA_LA == "la":
		thirdComp = 0
	else:
		raise ValueError("SA_LA must either be \"SA\" or \"LA\".")

	for i in range(0, maskSlice.shape[0]):
		pts = np.array([maskSlice[i, 1], maskSlice[i, 0], thirdComp])

		# In MATLAB, one would now do this elementwise multiplication: "pix = pts * pixScale". We do NOT
		# do this here because nibabel's img.affine matrix (recall, transform = img.affine) has this step
		# factored into it.

		tmp = transform @ np.append(pts, 1)
		contours[i, :, sliceIndex] = tmp[0:3]

# Helper function.
# "contours" is an m1 x 2 ndarray for some m1.
# Returns an m2 x 2 ndarray that is the result of cleaning up "contours".
def cleanContours(contours, downsample):
	if contours.shape[0] == 0:
		return np.array([])

	# Remove any points which lie outside of image. Note: simply calling np.nonzero() doesn't do the trick; np.nonzero()
	# returns a tuple, and we want the first element of that tuple.
	indicesToRemove = np.nonzero(contours[:, 0] < 0)[0]
	contours = ut.deleteHelper(contours, indicesToRemove, axis = 0)

	# Downsample.
	contours = contours[0:(contours.size - 1):downsample, :]

	# Remove points in the contours that are too far from the majority of the contour. (This probably happens due to holes in segmentation).
	return ut.removeFarPoints(contours)

# Helper struct for passing arguments to subplotHelperMulti().
class PlotSettings:
	def __init__(self, maskSlice, contours, color):
		self.maskSlice = maskSlice
		self.contours = contours
		self.color = color

# Helper function for plotting points from different datasets on the same plot.
def subplotHelperMulti(ax, plotSettingsList, title):
	ax.clear()  # The way that the timing of ax.clear() is handled (with the "clear = False" default arg to subplotHelper()) is somewhat messy, and can probably be improved somehow.
	for ps in plotSettingsList:
		subplotHelper(ax = ax, title = title, maskSlice = ps.maskSlice, contours = ps.contours, color = ps.color, clear = False)

# Helper function for creating subplots.
def subplotHelper(ax, title, maskSlice, contours, color, size = .5, clear = True):
	if clear:
		ax.clear()

	ax.set_title(title, loc = "center", fontsize = 10, wrap = True)
	ax.imshow(X = maskSlice, cmap = "gray")

	# Scatterplot. "s" is the size of the points in the scatterplot.
	return ax.scatter(x = contours[:, 0], y = contours[:, 1], s = size, c = color) # Most uses of this function will not make use of the output returned.

# Helper function used by masks2ContoursSA() and masks2ContoursLA().
# Returns (seg, transform, pixSpacing).
def readFromNIFTI(segName, frameNum):
	# Load NIFTI image and its header.
	img = nib.load(segName)
	img = MONAIutils.correct_nifti_header_if_necessary(img)
	hdr = img.header

   # Get the segmentation from the NIFTI file.
	seg = img.get_fdata()
	if seg.ndim > 3:  # if segmentation includes all time points
		seg = seg[:, :, :, frameNum].squeeze()

	# Get the 4x4 homogeneous affine matrix.
	transform = img.affine  # In the MATLAB script, we transposed the transform matrix at this step. We do not need to do this here due to how nibabel works.
	transform[0:2, :] = -transform[0:2, :] # This edit has to do with RAS system in Nifti

	# Initialize pixSpacing. In MATLAB, pix_spacing is info.PixelDimensions(1). After converting from 1-based
	# indexing to 0-based indexing, one might think that that means pixSpacing should be pixdim[0], but this is not the
	# case due to how nibabel stores NIFTI headers.
	pixdim = hdr.structarr["pixdim"][1:3 + 1]
	pixSpacing = pixdim[1]

	return (seg, transform, pixSpacing)

# maskSlice is a 2D ndarray, i.e it is a m x n ndarray for some m, n. This function returns a m x 2 ndarray, where
# each row in the array represents a point in the contour around maskSlice.
def getContoursFromMask(maskSlice, irregMaxSize):
	# First, clean the mask.

	# Remove irregularites with fewer than irregMaxSize pixels.Note, the "min_size" parameter to this function is
	# incorrectly named, and should really be called "max_size".
	maskSlice = morphology.remove_small_objects(np.squeeze(maskSlice), min_size=irregMaxSize, connectivity=2)  # Might have to come back and see if connectivity = 2 was the right choice

	# Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
	seed = np.copy(maskSlice)
	seed[1:-1, 1:-1] = maskSlice.max()
	maskSlice = np.squeeze(morphology.reconstruction(seed, maskSlice, method='erosion'))

	# The mask is now cleaned; we can use measure.find contours() now.

	# It is very important that .5 is used for the "level" parameter.
	#
	# From https://scikit-image.org/docs/0.7.0/api/skimage.measure.find_contours.html:
	# "... to find reasonable contours, it is best to find contours midway between the expected “light” and “dark” values.
	# In particular, given a binarized array, do not choose to find contours at the low or high value of the array. This
	# will often yield degenerate contours, especially around structures that are a single array element wide. Instead
	# choose a middle value, as above."
	lst = measure.find_contours(maskSlice, level = .5)

	# lst is a list of m_ x 2 ndarrays; the np.stack with axis = 0 used below converts
	# this list of ndarrays into a single ndarray by concatenating rows.

	# This "else" in the below is the reason this function is necessary; np.stack() does not accept an empty list.
	return np.vstack(lst) if not len(lst) == 0 else np.array([])


#########################################################################################

def finishUp(SAresults, LAcontours, frameNum, fldr, imgName):
	# Unpack input.
	(SAcontours, SAinserts) = SAresults

	# Get valve points.
	valves = getValvePoints(frameNum, fldr, imgName)

	# Initialize SAsliceIndices to hold the indices of the slices we will soon iterate over.
	# Short axis slices without any contours are omitted.
	SAsliceIndices = getSAsliceIndices(SAcontours)

	# Calculate apex using "method 2" from the MATLAB script.
	LA_LVepiContours = LAcontours["LVepi"]
	epiPts1 = np.squeeze(LA_LVepiContours[:, :, 0])
	epiPts2 = np.squeeze(LA_LVepiContours[:, :, 2])
	apex = mut.calcApex(epiPts1, epiPts2)

	# Plot the results.
	plotResults(SAsliceIndices, SAcontours, SAinserts, LAcontours, valves, apex)

	# Write the results to two text files.
	writeResults(frameNum, SAsliceIndices, SAcontours, SAinserts, LAcontours, valves, apex, fldr)

def getValvePoints(frameNum, fldr, imgName):
	# Get valve points for mitral valve (mv), tricuspid valve (tv), aortic valve (av), and pulmonary valve (pv).
	# Note: the valve points computed by this Python script are slightly different than those produced by MATLAB because
	# the interpolation function used in this code, np.interp(), uses a different interplation method than the MATLAB interp1().
	numFrames = nib.load(imgName).get_fdata().shape[3]  # all good
	(mv, tv, av, pv) = mut.manuallyCompileValvePoints(fldr, numFrames, frameNum)

	# Remove rows that are all zero from mv, tv, av, pv.
	mv = np.reshape(mv, (-1, 3))  # Reshape mv to have shape m x 3 for some m (the -1 indicates an unspecified value).
	mv = ut.removeZerorows(mv)
	tv = ut.removeZerorows(tv)
	av = ut.removeZerorows(av)
	pv = ut.removeZerorows(pv)
	return (mv, tv, av, pv)

def getSAsliceIndices(SAContours):
	SA_LVendoContours = SAContours["LVendo"]
	slicesToKeep = np.squeeze(SA_LVendoContours[0, 0, :]).astype(int)
	slicesToOmit1 = np.array(
		[i for i in range(SA_LVendoContours.shape[2]) if prepareContour(SA_LVendoContours, i).size == 0])
	slicesToOmit2 = np.logical_not(slicesToKeep)  # is 1 wherever slicesToKeep was 0.
	slicesToOmit = np.unique(np.concatenate((slicesToOmit1, slicesToOmit2)).astype(int))
	numSASlices = SA_LVendoContours.shape[2]
	SAslices = np.linspace(1, numSASlices, numSASlices)
	SAslices = ut.deleteHelper(SAslices, slicesToOmit, axis = 0).astype(int)
	return SAslices - 1  # convert to Python indexing


def plotResults(includedSlices, SAContours, SAinserts, LAContours, valves, apex):
	# Unwrap valve points.
	(mv, tv, av, pv) = valves

	# Plot short axis contours.
	fig, ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
	for i in includedSlices:
		LVendo = prepareContour(SAContours["LVendo"], i)
		LVepi = prepareContour(SAContours["LVepi"], i)
		RVFWendo = prepareContour(SAContours["RVFWendo"], i)
		RVFWepi = prepareContour(SAContours["RVFWepi"], i)
		RVsept = prepareContour(SAContours["RVsept"], i)
		RVinserts = prepareContour(SAinserts["RVinserts"], i)

		h0 = ax.scatter(LVendo[:, 0], LVendo[:, 1], LVendo[:, 2], marker = ".", color = "green")
		h1 = ax.scatter(LVepi[:, 0], LVepi[:, 1], LVepi[:, 2], marker = ".", color = "blue")
		h2 = ax.scatter(RVFWendo[:, 0], RVFWendo[:, 1], RVFWendo[:, 2], marker = ".", color = "red")
		h3 = ax.scatter(RVFWepi[:, 0], RVFWepi[:, 1], RVFWepi[:, 2], marker = ".", color = "blue")
		h4 = ax.scatter(RVsept[:, 0], RVsept[:, 1], RVsept[:, 2], marker = ".", color = "yellow")
		h5 = ax.scatter(RVinserts[:, 0], RVinserts[:, 1], RVinserts[:, 2], s = 50, color = "red")

	# Plot valve points.
	h6 = ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s = 50, color = "black")
	h7 = ax.scatter(av[:, 0], av[:, 1], av[:, 2], s = 50, color = "cyan")
	h8 = ax.scatter(tv[:, 0], tv[:, 1], tv[:, 2], s = 50, color = "magenta")

	# Plot apex.
	h9 = ax.scatter(apex[0], apex[1], apex[2], color = "black")

	# Plot long axis contours.
	numLASlices = LAContours["LVendo"].shape[2]
	for i in range(numLASlices):
		LVendo = prepareContour(LAContours["LVendo"], i)
		LVepi = prepareContour(LAContours["LVepi"], i)
		RVFWendo = prepareContour(LAContours["RVFWendo"], i)
		RVFWepi = prepareContour(LAContours["RVFWepi"], i)
		RVsept = prepareContour(LAContours["RVsept"], i)

		ax.scatter(LVendo[:, 0], LVendo[:, 1], LVendo[:, 2], marker = ".", color = "green")
		ax.scatter(LVepi[:, 0], LVepi[:, 1], LVepi[:, 2], marker = ".", color = "blue")
		ax.scatter(RVFWendo[:, 0], RVFWendo[:, 1], RVFWendo[:, 2], marker = "d", s = 30, color = "red") #using s = 30 for debug
		ax.scatter(RVFWepi[:, 0], RVFWepi[:, 1], RVFWepi[:, 2], marker = ".", color = "blue")
		ax.scatter(RVsept[:, 0], RVsept[:, 1], RVsept[:, 2], marker = "x", color = "yellow")

	ax.view_init()
	ax.legend((h0, h1, h2, h3, h4, h5, h6, h7, h8, h9),
			  ("LV endo", "Epi", "RVFW endo", "RVFW epi", "RV sept", "RV inserts",
			   "Mitral valve", "Aortic valve", "Tricuspid valve", "Apex"))
	plt.show()  # Must use plt.show() instead of fig.show(). Might have something to do with https://github.com/matplotlib/matplotlib/issues/13101#issuecomment-452032924


def writeResults(frameNum, includedSlices, SAContours, SAinserts, LAContours, valves, apex, fldr):
	# Unwrap valve points.
	(mv, tv, av, pv) = valves

	# Set up file writers.
	try:
		file1 = open(fldr + "GPFile_py.txt", "w", newline = "", encoding = "utf-8")
		file2 = open(fldr + "Case1_FR1_py.txt", "w", newline = "", encoding = "utf-8")
	except Exception as e:
		print(e)
		exit()

	writer1 = csv.writer(file1, delimiter = "\t")
	writer2 = csv.writer(file2, delimiter = " ")

	# TO-DO: precompute the results of prepareContour() and store them before plotting things, since results of
	# prepareContour() are used twice.

	def writePoint(point, j, name1, name2, weight = 1):
		writer2.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
						  name1, "{:d}".format(j + 1), "{:0.4f}".format(weight)])
		writer1.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
						 name2, "{:d}".format(j + 1), "{:0.4f}".format(weight), "{:d}".format(frameNum)])

	def writeContour(mask, i, j, name1, name2, rvi_weights = False):
		if rvi_weights:
			RVInsertsWeights = SAinserts["RVinsertsWeights"]
			multiIndex = np.unravel_index(i, RVInsertsWeights.shape, order = "F")
			weight = RVInsertsWeights[multiIndex]
		else:
			weight = 1

		for k in range(0, mask.shape[0]):
			writePoint(mask[k, :], j, name1, name2, weight)

	# Write short axis contours to text files
	writer1.writerow(["x", "y", "z", "contour type", "slice", "weight", "time frame"])
	for j, i in enumerate(
			includedSlices):  # i will be the ith included slice, and j will live in range(len(includedSlices))
		# LV endo
		LVendo = prepareContour(SAContours["LVendo"], i)
		writeContour(LVendo, i, j, "saendocardialContour", "SAX_LV_ENDOCARDIAL")

		# RV free wall
		RVFW = prepareContour(SAContours["RVFWendo"], i)
		writeContour(RVFW, i, j, "RVFW", "SAX_RV_FREEWALL")

		# Epicardium
		LVepi = prepareContour(SAContours["LVepi"], i)
		RVepi = prepareContour(SAContours["RVFWepi"], i)
		epi = np.vstack((LVepi, RVepi))
		writeContour(epi, i, j, "saepicardialContour", "SAX_LV_EPICARDIAL")

		# RV septum
		RVsept = prepareContour(SAContours["RVsept"], i)
		writeContour(RVsept, i, j, "RVS", "SAX_RV_SEPTUM")

		# RV inserts
		RVI = prepareContour(SAinserts["RVinserts"], i)
		writeContour(RVI, i, j, "RV_insert", "RV_INSERT", rvi_weights = True)

	# Now do long axis contours
	numLASlices = LAContours["LVendo"].shape[2]
	last_SA_j = len(includedSlices) - 1
	first_LA_j = last_SA_j + 1
	LA_i_range = range(numLASlices)  # we will use LA_i_range and LA_j_range after the loop
	LA_j_range = range(first_LA_j, first_LA_j + numLASlices)

	for i, j in zip(LA_i_range, LA_j_range):
		# LV endo
		LVendo = prepareContour(LAContours["LVendo"], i)
		writeContour(LVendo, i, j, "saendocardialContour", "LAX_LV_ENDOCARDIAL")

		# RV free wall
		RVFW = prepareContour(LAContours["RVFWendo"], i)
		writeContour(LVendo, i, j, "RVFW", "LAX_RV_FREEWALL")

		# Epicardium
		LVepi = prepareContour(LAContours["LVepi"], i)
		RVepi = prepareContour(LAContours["RVFWepi"], i)
		epi = np.vstack((LVepi, RVepi))
		writeContour(epi, i, j, "saepicardialContour", "LAX_LV_EPICARDIAL")

		# RV septum
		RVsept = prepareContour(LAContours["RVsept"], i)  # not doing isempty check
		writeContour(RVsept, i, j, "RVS", "LAX_RV_SEPTUM")

		# Valves (tricuspid, mitral, aortic, pulmonary)
		writeContour(tv, i, j, "Tricuspid_Valve", "TRICUSPID_VALVE")
		writeContour(mv, i, j, "BP_point", "MITRAL_VALVE")
		writeContour(av, i, j, "Aorta", "AORTA_VALVE")
		writeContour(pv, i, j, "Pulmonary", "PULMONARY_VALVE")

	# Finally, write out the apex
	writePoint(apex, LA_j_range[-1], "LA_Apex_Point", "APEX_POINT")

def prepareContour(mask, sliceIndex):
	result = np.squeeze(mask[:, :, sliceIndex])
	return ut.removeZerorows(result)