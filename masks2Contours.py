import sys
sys.path.append("masks2ContoursScripts")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage import measure
from skimage import morphology

import MONAIutils
import masks2ContoursUtil as ut
from SelectFromCollection import SelectFromCollection

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
        inputsList = (LVendo[:, :, i], LVepi[:, :, i], RVendo[:, :, i], transform, pixSpacing) #Note the last 3 args are the same every iteration.
        outputsList = (LVendoContours, LVepiContours, RVseptContours, RVFWendoContours, RVFWepiContours, RVinserts)
        slice2ContoursPt1(inputsList, outputsList, config, i, "SA", MPL_objs = None)

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

def masks2ContoursLA(LA_segs, frameNum, numSlices, config, MPL_objs):
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

    # Loop over slices and get contours one slice at a time.
    for i in range(0, numSlices):
        inputsList = (LVendoList[i], LVepiList[i], RVendoList[i], transformList[i], pixSpacingList[i])
        outputsList = (LVendoContours, LVepiContours, RVseptContours, RVFWendoContours, RVFWepiContours)
        slice2ContoursPt1(inputsList, outputsList, config, i, "LA", MPL_objs)

    # Return a dictionary.
    return {"LVendo": LVendoContours,
            "LVepi": LVepiContours,
            "RVFWendo": RVFWendoContours,
            "RVFWepi": RVFWepiContours,
            "RVsept": RVseptContours}

def slice2ContoursPt1(inputsList, outputsList, config, sliceIndex, SA_LA, MPL_objs):
    # Check validity of SA_LA.
    SA_LA = SA_LA.lower()
    if not(SA_LA == "sa" or SA_LA == "la"):
        raise ValueError("SA_LA must either be \"SA\" or \"LA\".")

    # Unpack tuples passed in as input.
    (LVendo, LVepi, RVendo, transform, pixSpacing) = inputsList

    # Still unpacking stuff...
    if SA_LA == "sa":
        (LVendoContours, LVepiContours, RVseptContours, RVFWendoContours, RVFWepiContours, RVinserts) = outputsList
    else: # SA_LA == "la"
        (LVendoContours, LVepiContours, RVseptContours, RVFWendoContours, RVFWepiContours) = outputsList

    # Get contours for each slice in the different regions of the heart. In the following variable names,
    # "CS" stands for "contour slice".
    LVendoCS = getContoursFromMask(LVendo, irregMaxSize = 20)
    LVepiCS = getContoursFromMask(LVepi, irregMaxSize = 20)
    RVendoCS = getContoursFromMask(RVendo, irregMaxSize = 20)

    # Load MATLAB variables if config indicates that we should do so.
    LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
    LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
    RVendoIsEmpty = RVendoCS is None or RVendoCS.size == 0

    if not LVendoIsEmpty:
        LVendoCS[:, [0, 1]] = LVendoCS[:, [1, 0]]
    if not LVepiIsEmpty:
        LVepiCS[:, [0, 1]] = LVepiCS[:, [1, 0]]
    if not RVendoIsEmpty:
        RVendoCS[:, [0, 1]] = RVendoCS[:, [1, 0]]

    # Differentiate contours for RVFW (free wall) and RVS (septum).
    # Recall "CS" stands for "contour slice".
    [RVseptCS, ia, ib] = ut.sharedRows(LVepiCS, RVendoCS)
    RVFW_CS = RVendoCS
    RVFW_CS = ut.deleteHelper(RVFW_CS, ib, axis = 0)  # In RVFW_CS, delete the rows with index ib.

    # Remove RVS contour points from LV epi contour
    LVepiCS = ut.deleteHelper(LVepiCS, ia, axis = 0)  # In LVepiCS, delete the rows with index ia.

    # If doing long axis, remove line segments at base which are common to LVendoCS and LVepiCS.
    if SA_LA == "la":
        [wont_use_this_var, ia, ib] = ut.sharedRows(LVendoCS, LVepiCS)
        LVendoCS = ut.deleteHelper(LVendoCS, ia, axis = 0)
        LVepiCS = ut.deleteHelper(LVepiCS, ib, axis = 0)

    # Update the "emptiness" status of our contour slices.
    LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
    LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
    RVendoIsEmpty = RVFW_CS is None or RVFW_CS.size == 0

    # LV endo
    if not LVendoIsEmpty:
        LVendoCS = cleanContours(LVendoCS, config.downsample)
        contoursToImageCoords(LVendoCS, transform, sliceIndex, LVendoContours, SA_LA)  # This call writes to "endoLVContours".

    # LV epi
    if not LVepiIsEmpty:
        LVepiCS = cleanContours(LVepiCS, config.downsample)
        contoursToImageCoords(LVepiCS, transform, sliceIndex, LVepiContours, SA_LA)  # This call writes to "epiLVContours".

    # RV
    if not RVendoIsEmpty:
        RVFW_CS = cleanContours(RVFW_CS, config.downsample)
        RVseptCS = cleanContours(RVseptCS, config.downsample)

        # If doing short axis, get the indices corresponding to insertion points in the RV.
        if SA_LA == "sa":
            tmpRV_insertIndices = ut.getRVinsertIndices(RVFW_CS)

        RVEndoIsEmptyAfterCleaning = (RVFW_CS is None or RVseptCS is None) or (RVFW_CS.size == 0 or RVseptCS.size == 0)

        # Wrap up stuff we've computed so that it can be passed to slice2ContoursPt2() in the below.
        RVdata = (RVseptCS, RVFW_CS, RVseptContours, RVFWendoContours, RVFWepiContours)
        if SA_LA == "sa":
            RVdata_SAonly = (tmpRV_insertIndices, RVinserts)
        geoData = (transform, pixSpacing)
        otherData = (config, SA_LA)

        if SA_LA == "sa":
            pt2Data = {"RVdata": RVdata, "RVdata_SAonly": RVdata_SAonly, "geoData": geoData, "otherData": otherData}
        else:  # SA_LA == "la"
            pt2Data = {"RVdata": RVdata, "geoData": geoData, "otherData": otherData}

        # If doing long axis, remove basal line segment in RVFW LA contour.
        if SA_LA == "la" and not RVEndoIsEmptyAfterCleaning:
            # Set up the scatter plot that will be passed to the lasso selector.
            figI, axI = plt.subplots() # I for "inspection"
            title = "Select points you would like to remove. Click and drag to lasso select.\n The zoom tool may also be helpful."
            pts = subplotHelper(axI, title = title, maskSlice = np.squeeze(RVendo), contours = RVFW_CS, color = "green", size = 20)

            # Create the lasso selector.
            lassoSelector = SelectFromCollection(figI, axI, pts)

            # Store the stuff we've computed as a member of the lassoSelector; the callback function
            # lassoSelector.key_press() will call slice2ContoursPt2(self.pt2Data, self.sliceIndex).
            lassoSelector.pt2Data = pt2Data
            lassoSelector.sliceIndex = sliceIndex

            # This causes plot-with-interactive-lasso-selector to appear. When the user presses Enter,
            # slice2ContoursPt2() is called by the callback function lassoSelector.keypress().
            (mgr, widg) = MPL_objs
            widg.lasso = lassoSelector # Prevents lassoSelector from getting garbage collected
            mgr.win.createDock("MPL Widget", widg)
            lassoSelector.parent_widg = mgr.win.dockWidgets[-1].parent() # dock widget contains `widg`, parent of that is dock itself
        else:
            slice2ContoursPt2(pt2Data, sliceIndex)

def slice2ContoursPt2(pt2Data, sliceIndex):
    # Unpack input.
    (RVseptCS, RVFW_CS, RVseptContours, RVFWendoContours, RVFWepiContours) = pt2Data["RVdata"]
    (transform, pixSpacing) = pt2Data["geoData"]
    (config, SA_LA) = pt2Data["otherData"]
    if SA_LA == "sa":
        (tmpRV_insertIndices, RVinserts) = pt2Data["RVdata_SAonly"]

    # Write to RVseptContours and RVFWendoContours.
    contoursToImageCoords(RVseptCS, transform, sliceIndex, RVseptContours, SA_LA)
    contoursToImageCoords(RVFW_CS, transform, sliceIndex, RVFWendoContours, SA_LA)

    # If doing short axis, save RV insert coordinates.
    if SA_LA == "sa" and (tmpRV_insertIndices is not None and tmpRV_insertIndices.size != 0):
        RVinserts[:, :, sliceIndex] = RVFWendoContours[tmpRV_insertIndices[0:1 + 1], :, sliceIndex]

    # Calculate RV epicardial wall by applying a thickness to RV endocardium.
    if SA_LA == "sa":
        RVEndoNormals = ut.lineNormals2D(RVFW_CS)
        RVepiSC = RVFW_CS - np.ceil(config.rvWallThickness / pixSpacing) * RVEndoNormals \
            if RVFW_CS.size != 0 and RVEndoNormals.size != 0 else np.array([])
        contoursToImageCoords(RVepiSC, transform, sliceIndex, RVFWepiContours, "SA")

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