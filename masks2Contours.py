import numpy as np
import nibabel as nib
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt
import scipy.io as sio
import masks2ContoursUtil as ut

# Note: axes are swapped when contours loaded with Python, for some reason. This is accounted for in this function, but
# is still something to investigate.

def masks2ContoursSA(segName, imgName, resultsDir, frameNum, config):
    (seg, transform, pixScale, pixSpacing) = readFromNIFTI(segName, frameNum)
    numSlices = seg.shape[2]

    # Separate out LV endo, LV epi, and RV endo segmentations.
    endoLV = seg == 1
    epiLV = (seg <= 2) * (seg > 0)
    endoRV = seg == 3

    # Initialize variables.
    ub = config.upperBdNumContourPts
    endoLVContours = np.zeros([ub, 3, numSlices])
    epiLVContours = np.zeros([ub, 3, numSlices])
    endoRVFWContours = np.zeros([ub, 3, numSlices])
    epiRVFWContours = np.zeros([ub, 3, numSlices])
    RVSContours = np.zeros([ub, 3, numSlices])
    RVInserts = np.zeros([2, 3, numSlices])

    # Set up the machinery that will handle plotting.
    if config.PLOT:
        fig, axs = plt.subplots(1, 3)  # A Figure with a 1 x 3 grid of Axes.
        for ax in axs:
            ax.axis("equal")
        fig.tight_layout()  # might use this: https://stackoverflow.com/questions/8802918/my-matplotlib-title-gets-cropped

    # Load some MATLAB variables, for debug.
    removedPointsAll = sio.loadmat(resultsDir + "removedPointsAll.mat")["removedPointsAll"]
    endoRVFWContours_mat = sio.loadmat(resultsDir + "endoRVFWContours.mat")["endoRVFWContours"]

    ###############################################################
    # Loop over slices and get contours one slice at a time
    ###############################################################
    for i in range(0, numSlices):
        endoLVCurrent = endoLV[:, :, i]
        epiLVCurrent = epiLV[:, :, i]
        endoRVCurrent = endoRV[:, :, i]
        transformCurrent = transform # constant
        pixScaleCurrent = pixScale # constant
        pixSpacingCurrent = pixSpacing # constant

        # Get contours from masks.
        tmp_endoLV = getContoursFromMask(endoLVCurrent, irregMaxSize = 50)
        tmp_epiLV = getContoursFromMask(epiLVCurrent, irregMaxSize = 50)
        tmp_endoRV = getContoursFromMask(endoRVCurrent, irregMaxSize = 50)

        ##########################################################################################################
        # Remove this section after done testing. See https://docs.scipy.org/doc/scipy/reference/tutorial/io.html.
        ##########################################################################################################
        if config.LOAD_MATLAB_VARS:
            # Load MATLAB variables.
            endoLVmask_mat = sio.loadmat(resultsDir + "endoLVmask_slices\\endoLVmask_slice_" + str(i + 1))["endoLVmask"]
            epiLVmask_mat = sio.loadmat(resultsDir + "epiLVmask_slices\\epiLVmask_slice_" + str(i + 1))["epiLVmask"]
            endoRVmask_mat = sio.loadmat(resultsDir + "endoRVmask_slices\\endoRVmask_slice_" + str(i + 1))["endoRVmask"]

            tmp_endoLV_mat = sio.loadmat(resultsDir + "tmp_endoLV_slices\\tmp_endoLV_slice_" + str(i + 1))["tmp_endoLV"]
            tmp_epiLV_mat = sio.loadmat(resultsDir + "tmp_epiLV_slices\\tmp_epiLV_slice_" + str(i + 1))["tmp_epiLV"]
            tmp_endoRV_mat = sio.loadmat(resultsDir + "tmp_endoRV_slices\\tmp_endoRV_slice_" + str(i + 1))["tmp_endoRV"]

            # Use the MATLAB variables instead of the Python ones.
            endoLVmask = endoLVmask_mat
            epiLVmask = epiLVmask_mat
            endoRVmask = endoRVmask_mat

            tmp_endoLV = tmp_endoLV_mat
            tmp_epiLV = tmp_epiLV_mat
            tmp_endoRV = tmp_endoRV_mat

            # Ensure that when a slice is empty, it has a ".shape" of (0, )
            matVars = [endoLVmask, epiLVmask, endoRVmask, tmp_endoLV, tmp_epiLV, tmp_endoRV]
            for x in range(len(matVars)):
                if matVars[x].size == 0:
                    matVars[x] = np.array([])
            endoLVmask, epiLVmask, endoRVmask, tmp_endoLV, tmp_epiLV, tmp_endoRV = matVars

        # Differentiate contours for RVFW (free wall) and RVS (septum)
        [tmp_RVS, ia, ib] = ut.sharedRows(tmp_epiLV, tmp_endoRV)
        tmp_RVFW = tmp_endoRV
        tmp_RVFW = ut.deleteHelper(tmp_RVFW, ib, axis = 0) #In tmpRVFW, delete the rows with index ib.

        # Remove RVS contour points from LV epi contour
        tmp_epiLV = ut.deleteHelper(tmp_epiLV, ia, axis = 0) #In tmp_epiLV, delete the rows with index ia.

        # These variables are for readability.
        LVEndoIsEmpty = tmp_endoLV is None or tmp_endoLV.size == 0
        LVEpiIsEmpty = tmp_epiLV is None or tmp_epiLV.size == 0
        RVEndoIsEmpty = tmp_RVFW is None or tmp_RVFW.shape[0] <= 6

        #LV endo
        if not LVEndoIsEmpty:

            tmp_endoLV = cleanContours(tmp_endoLV, config.downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            LVEndoIsEmptyAfterCleaning = tmp_endoLV is None or tmp_endoLV.size == 0
            if config.PLOT and not LVEndoIsEmptyAfterCleaning:
                subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLVCurrent),
                              contours = tmp_endoLV, color = "red", swap = not config.LOAD_MATLAB_VARS) # Python contours on Python mask

                # subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLV[:, :, i]),
                # contours = np.squeeze(tmp_endoLVAll[:, :, i]), color = "green") # MATLAB contours on Python mask

                contoursToImageCoords(tmp_endoLV, transformCurrent, pixScaleCurrent, i, endoLVContours, "SA")  # This call writes to "endoLVContours".

        #LV epi
        if not LVEpiIsEmpty:

            # In this case, we do basically the same thing as above, except with tmp_epiLV, epiLVmask, and epiLVContours instead of
            # tmp_endoLV, endoLVmask, and endoLVContours. A function is not used to generalize the work done by this and
            # the previous case because, since the next case doesn't exactly follow the same format, writing such a function
            # would do more to obfusticate than to simplify).

            tmp_epiLV = cleanContours(tmp_epiLV, config.downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            LVEpiIsEmptyAfterCleaning = tmp_endoLV is None or tmp_endoLV.size == 0
            if config.PLOT and not LVEpiIsEmptyAfterCleaning:
                subplotHelper(axs[1], title = "LV Epicardium", maskSlice = np.squeeze(epiLVCurrent),
                              contours = tmp_epiLV, color = "blue", swap = not config.LOAD_MATLAB_VARS) #Python contours on Python mask

                # subplotHelper(axs[1], title = "LV Epicardium", maskSlice = np.squeeze(epiLV[:, :, i]),
                # contours = np.squeeze(tmp_epiLVAll[:, :, i]), color = "cyan") # MATLAB contours on Python mask

                contoursToImageCoords(tmp_epiLV, transformCurrent, pixScaleCurrent, i, epiLVContours, "SA")  # This call writes to "epiLVContours".

        # RV endo
        if not RVEndoIsEmpty:

            tmp_RVFW = cleanContours(tmp_RVFW, config.downsample)
            tmpRV_insertIndices = ut.getRVinsertIndices(tmp_RVFW)
            tmp_RVS = cleanContours(tmp_RVS, config.downsample) #possibly switch this and the previous line for organization

            RVEndoIsEmptyAfterCleaning = (tmp_RVFW is None or tmp_RVS is None) or (tmp_RVFW.size == 0 or tmp_RVS.size == 0)
            if config.PLOT and not RVEndoIsEmptyAfterCleaning:
                ps1 = plotSettings(maskSlice = np.squeeze(endoRVCurrent), contours = tmp_RVFW, color = "green", swap = not config.LOAD_MATLAB_VARS)
                ps2 = plotSettings(maskSlice = np.squeeze(endoRVCurrent), contours = tmp_RVS, color = "yellow", swap = not config.LOAD_MATLAB_VARS)
                subplotHelperMulti(axs[2], plotSettingsList = [ps1, ps2], title = "RV Endocardium")

            contoursToImageCoords(tmp_RVS, transformCurrent, pixScaleCurrent, i, RVSContours, "SA")
            contoursToImageCoords(tmp_RVFW, transformCurrent, pixScaleCurrent, i, endoRVFWContours, "SA")

            # Save RV insert coordinates
            if tmpRV_insertIndices is not None and tmpRV_insertIndices.size != 0:
                RVInserts[:, :, i] = endoRVFWContours[tmpRV_insertIndices[0:1 + 1], :, i]

        # Calculate RV epicardial wall by applying a thickness to RV endocardium

        # RV epicardium
        RVEndoNormals = ut.lineNormals2D(tmp_RVFW)
        tmp_epiRV = tmp_RVFW - np.ceil(config.rvWallThickness/pixSpacingCurrent) * RVEndoNormals if tmp_RVFW.size != 0 and RVEndoNormals.size != 0 else np.array([])
        contoursToImageCoords(tmp_epiRV, transformCurrent, pixScaleCurrent, i, epiRVFWContours, "SA")

        # Show the figure for .5 seconds if config.PLOT is True.
        if config.PLOT and not (LVEndoIsEmpty or LVEpiIsEmpty or RVEndoIsEmpty):
            fig.show()
            while True:
                 #if plt.waitforbuttonpress(): break
                 if plt.waitforbuttonpress(timeout = .5) is None: break

        # In MATLAB, "clearvars tmp*" is called at this point.

    ##############################################
    # Calculate weights for RV insertion points
    ##############################################

    # Initialise variable to store error
    RVInsertsWeights = np.zeros((2, numSlices))

    # Loop through anterior (1) and posterior (2) sides for fitting a line
    for i in range(0, 2):

        # In the MATLAB script, the following step essentially amounted to horizontally stacking _inserts1^T and _inserts2^T. (^T is matrix transpose).
        # Since transposing a 1 x n ndarray produces a 1 x n ndarray, rather than a n x 1 ndarray (this happens because numpy
        # tries to make matrix transposition more efficent, since tranposition is a very predictable shuffle of old data)
        # we do a different but equivalent operation here. That is, we take the transpose of the ndarray obtained by stacking
        # _inserts2 below _inserts1.
        _inserts1 = np.squeeze(RVInserts[i, :, :])
        _inserts2 = np.linspace(0, numSlices - 1, numSlices) #do we need to use 0, slices - 1 for first two args of linspace?
        inserts = np.vstack((_inserts1, _inserts2))
        inserts = inserts.transpose()
        inserts = inserts[np.all(inserts[:, 0:2], 1), :] # Get rid of rows with zeros in the first three columns.

        points = inserts[:, 0:3]
        indices = inserts[:, 3].astype(int)

        # Sort RV insert points by error, and save the normalized error in RVInsertsWeights.
        err = ut.fitLine3D(points)
        RVInsertsWeights[i, indices] = np.abs(err)/np.max(np.abs(err))

        return (endoLVContours, epiLVContours, endoRVFWContours, epiRVFWContours, RVSContours, RVInserts, RVInsertsWeights)

def masks2ContoursLA(LA_names, LA_segs, resultsDir, frameNum, config):
    numSlices = len(LA_names)

    # Precompute endoLV, epiLV, endoRV for each slice. Also precompute transform, pixScale, and pix_spacing for each slice.
    endoLVList = []
    epiLVList = []
    endoRVList = []
    transformList = []
    pixScaleList = []
    pixSpacingList = []
    for i in range(0, numSlices):
        (seg, transform, pixScale, pixSpacing) = readFromNIFTI(LA_segs[i], frameNum)
        endoLVList.append(np.squeeze(seg == 1))
        epiLVList.append(np.squeeze((seg <= 2) * (seg > 0)))
        endoRVList.append(np.squeeze(seg == 3))
        transformList.append(transform)
        pixScaleList.append(pixScale)
        pixSpacingList.append(pixSpacing)

    # Initialize variables.
    ub = config.upperBdNumContourPts
    endoLVContours = np.zeros([ub, 3, numSlices])
    epiLVContours = np.zeros([ub, 3, numSlices])
    endoRVFWContours = np.zeros([ub, 3, numSlices])
    epiRVFWContours = np.zeros([ub, 3, numSlices])
    RVSContours = np.zeros([ub, 3, numSlices])

    # Set up the machinery that will handle plotting.
    if config.PLOT:
        fig, axs = plt.subplots(1, 3)  # A Figure with a 1 x 3 grid of Axes.
        for ax in axs:
            ax.axis("equal")
        fig.tight_layout()  # might use this: https://stackoverflow.com/questions/8802918/my-matplotlib-title-gets-cropped

    for i in range(0, numSlices):
        # Get name of current LA view
        # !! This is unique to LA !!
        tmp = LA_segs[i].split("\\")
        tmpLA = tmp[-1].split("_")

        # Unpack precomputed things.
        endoLVCurrent = endoLVList[i]
        epiLVCurrent = epiLVList[i]
        endoRVCurrent = endoRVList[i]
        transformCurrent = transformList[i]
        pixScaleCurrent = pixScaleList[i]
        pisSpacingCurrent = pixSpacingList[i]

        # Get contours from masks.
        tmp_endoLV = getContoursFromMask(endoLVCurrent, irregMaxSize = 20)
        tmp_epiLV = getContoursFromMask(epiLVCurrent, irregMaxSize = 20)
        tmp_endoRV = getContoursFromMask(endoRVCurrent, irregMaxSize = 20)

        # Differentiate contours for RVFW (free wall) and RVS (septum).
        [tmp_RVS, ia, ib] = ut.sharedRows(tmp_epiLV, tmp_endoRV)
        tmp_RVFW = tmp_endoRV
        tmp_RVFW = ut.deleteHelper(tmp_RVFW, ib, axis = 0)  # In tmpRVFW, delete the rows with index ib.

        # Remove RVS contour points from LV epi contour
        tmp_epiLV = ut.deleteHelper(tmp_epiLV, ia, axis = 0)  # In tmp_epiLV, delete the rows with index ia.

        # Remove line segments at base which are common to endoLV and epiLV
        # !! this is unique to LA !!
        [wont_use_this_var, ia, ib] = ut.sharedRows(tmp_endoLV, tmp_epiLV)
        tmp_endoLV = ut.deleteHelper(tmp_endoLV, ia, axis = 0)
        tmp_epiLV = ut.deleteHelper(tmp_epiLV, ib, axis = 0)

        # These variables are for readability.
        LVEndoIsEmpty = tmp_endoLV is None or np.max(tmp_endoLV.shape) <= 2
        LVEpiIsEmpty = tmp_epiLV is None or np.max(tmp_epiLV.shape) <= 2
        RVEndoIsEmpty = tmp_RVFW is None or tmp_RVFW.size == 0

        # LV endo
        if not LVEndoIsEmpty:

            tmp_endoLV = cleanContours(tmp_endoLV, config.downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            LVEndoIsEmptyAfterCleaning = tmp_endoLV is None or tmp_endoLV.size == 0
            if config.PLOT and not LVEndoIsEmptyAfterCleaning:
                # !! unique to LA !!:
                # "maskSlice = np.squeeze(endoLV)" instead of "maskSlice = np.squeeze(endoLV[:, :, i]) #
                # swap = True instead of swap = not LOAD_MATLAB_VARS
                subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLVCurrent),
                              contours = tmp_endoLV, color = "red",
                              swap = True)  # Python contours on Python mask

                # subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLV[:, :, i]),
                # contours = np.squeeze(tmp_endoLVAll[:, :, i]), color = "green") # MATLAB contours on Python mask

                contoursToImageCoords(tmp_endoLV, transformCurrent, pixScaleCurrent, i, endoLVContours, "LA")  # This call writes to "endoLVContours".

        # Show the figure for .5 seconds if config.PLOT is True.
        if config.PLOT and not (LVEndoIsEmpty or LVEpiIsEmpty or RVEndoIsEmpty):
            fig.show()
            while True:
                #if plt.waitforbuttonpress(): break
                if plt.waitforbuttonpress(timeout = .5) is None: break

# Helper function for converting contours to an image coordinate system. This function writes to "contours".
def contoursToImageCoords(tmp_contours, transform, pixScale, sliceIndex, contours, SA_LA):
    SA_LA = SA_LA.lower()
    if SA_LA == "sa":
        thirdComp = sliceIndex
    elif SA_LA == "la":
        thirdComp = 0
    else:
        raise ValueError("SA_LA must either be ``SA'' or ``LA''.")

    for i in range(0, tmp_contours.shape[0]):
        pix = np.array([tmp_contours[i, 1], tmp_contours[i, 0], thirdComp]) * pixScale
        tmp = transform @ np.append(pix, 1)
        contours[i, :, sliceIndex] = tmp[0:3]

# Helper function used in finalizeContour.
# "contours" is an m1 x 2 ndarray for some m1.
# Returns an m2 x 2 ndarray that is the result of cleaning up "contours".
def cleanContours(contours, downsample):
    if contours.shape[0] == 0:
        return np.array([])

    # Remove any points which lie outside of image. Note: simply calling np.nonzero() doesn't do the trick; np.nonzero()
    # returns a tuple, and we want the first element of that tuple.
    indicesToRemove = np.nonzero(contours[:, 0] < 0)[0]
    contours = ut.deleteHelper(contours, indicesToRemove, axis = 0) #COME BACK

    # Downsample.
    contours = contours[0:(contours.size - 1):downsample, :]

    # Remove points in the contours that are too far from the majority of the contour. (This probably happens due to holes in segmentation).
    return ut.removeFarPoints(contours)

class plotSettings:
    def __init__(self, maskSlice, contours, color, swap = False):
        self.maskSlice = maskSlice
        self.contours = contours
        self.color = color
        self.swap = swap

def subplotHelperMulti(ax, plotSettingsList, title):
    ax.clear()  # The way that the timing of ax.clear() is handled (with the "clear = False" default arg to subplotHelper()) is somewhat messy, and can probably be improved somehow.
    for ps in plotSettingsList:
        subplotHelper(ax, title, ps.maskSlice, ps.contours, ps.color, ps.swap, clear = False)

# Helper function for creating subplots.
def subplotHelper(ax, title, maskSlice, contours, color, swap = False, clear = True):
    if clear:
        ax.clear()

    ax.set_title(title, loc = "center", fontsize = 10, wrap = True)
    ax.imshow(X = maskSlice, cmap = "gray")

    # Rotate and flip the contours if necessary.
    xx = contours[:, 0]
    yy = contours[:, 1]
    if swap: # Swap xx and yy.
        xx, yy = yy, xx

    # Scatterplot. "s" is the size of the points in the scatterplot.
    ax.scatter(x = xx, y = yy, s = .5, c = color)

# Helper function used by masks2ContoursSA() and masks2ContoursLA().
# If returnAll = True, returns (seg, transform, pixScale, pixSpacing). Otherwise, returns seg.
def readFromNIFTI(segName, frameNum, returnAll = True):
    # Load short axis segmentations and header info
    _seg = nib.load(segName)
    seg = _seg.get_fdata()
    info = _seg.header

    if seg.ndim > 3:  # if segmentation includes all time points
        seg = seg[:, :, :, frameNum].squeeze()

    if returnAll:
        # Obtain the 4x4 homogeneous affine matrix
        transform = _seg.affine  # In the MATLAB script, we transposed the transform matrix at this step. We do not need to do this here due to how nibabel works.
        transform[:2, :] = transform[:2, :] * -1  # This edit has to do with RAS system in Nifti files

        # Initialize pixScale. In the MATLAB script, pixScale was a column vector. Here, it will be a row vector.
        pixdim = info.structarr["pixdim"][1:3 + 1]
        if np.isclose(np.linalg.norm(transform[:, 0]), 1):
            pixScale = pixdim  # Here we are manually going into the header file's data. This is somewhat dangerous- maybe it can be done a better way?
        else:
            pixScale = np.array([1, 1, 1])

        # Initialize some more things.
        pixSpacing = pixdim[0]

        return (seg, transform, pixScale, pixSpacing)
    else:
        return seg

# mask2D is a 2D ndarray, i.e it is a m x n ndarray for some m, n. This function returns a m x 2 ndarray, where
# each row in the array represents a point in the contour around mask2D.
def getContoursFromMask(mask2D, irregMaxSize):
    mask2D = np.squeeze(cleanMask(mask2D, irregMaxSize))
    # It is very important that .5 is used for the "level" parameter.
    #
    # From https://scikit-image.org/docs/0.7.0/api/skimage.measure.find_contours.html:
    # "... to find reasonable contours, it is best to find contours midway between the expected “light” and “dark” values.
    # In particular, given a binarized array, do not choose to find contours at the low or high value of the array. This
    # will often yield degenerate contours, especially around structures that are a single array element wide. Instead
    # choose a middle value, as above."
    lst = measure.find_contours(mask2D, level = .5)

    # lst is a list of m_ x 2 ndarrays; the np.stack with axis = 0 used below converts
    # this list of ndarrays into a single ndarray by concatenating rows.

    # This "else" in the below is the reason this function is necessary; np.stack() does not accept an empty list.
    return np.vstack(lst) if not len(lst) == 0 else np.array([])

# Helper function for removing irregularities from masks.
def cleanMask(mask, irregMaxSize):
    # Remove irregularites with fewer than irregMaxSize pixels.Note, the "min_size" parameter to this function is
    # incorrectly named, and should really be called "max_size".
    morphology.remove_small_objects(np.squeeze(mask), min_size = irregMaxSize, connectivity = 2) # Might have to come back and see if connectivity = 2 was the right choice

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    return morphology.reconstruction(seed, mask, method='erosion')