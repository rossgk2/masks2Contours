import numpy as np
import nibabel as nib
from skimage import morphology
from skimage import measure
import matplotlib.axes as ax
import matplotlib.pyplot as plt
import scipy.io as sio
import masks2contours_util as ut

# TO-DO: Run Renee's new examples and compare contours
# TO-DO: Are axes getting swapped somehow?
# TO-DO: Finish RV section of this file.
# TO-DO: Finish this file.
#
# change point size in plots
# I think some "far points" are showing up in MATLAB plot because removeFarPoints isn't called on the MATLAB data.

def masks2contoursSA_manual(segName, imgName, resultsDir, frameNum, PLOT):
    rv_wall = 3  # RV wall thickness, in [mm] (don't have contours)
    downsample = 3  # e.g. 3 ==> downsample by taking every third point

    # Load short axis segmentations and header info
    _seg = nib.load(segName)
    seg = _seg.get_fdata()
    info = _seg.header

    if seg.ndim > 3: # if segmentation includes all time points
        seg = seg[:, :, :, frameNum].squeeze()

    # Obtain the 4x4 homogeneous affine matrix
    transform = _seg.affine  # In the MATLAB script, we transposed the transform matrix at this step. We do not need to do this here due to how nibabel works.
    transform[:2, :] = transform[:2, :] * -1  # This edit has to do with RAS system in Nifti files

    # Initialize pix_scale. In the MATLAB script, pix_scale was a column vector. Here, it will be a row vector.
    pixdim = info.structarr["pixdim"][1:3 + 1]
    if np.isclose(np.linalg.norm(_seg.affine[:, 0]), 1):
        pix_scale = pixdim  # Here we are manually going into the header file's data. This is somewhat dangerous- maybe it can be done a better way?
    else:
        pix_scale = np.array([1, 1, 1])

    # Pixel spacing
    pixSpacing = pixdim[0]

    # Number of slices
    slices = seg.shape[2]

    # Separate out LV endo, LV epi, and RV endo segmentations
    endoLV = seg == 1
    epiLV = (seg <= 2) * (seg > 0)
    endoRV = seg == 3

    # Initialise variables
    large_num = 200 # Use a number that is greater than or equal to the number of contour points.
    endoLVContours = np.zeros([large_num, 3, slices])
    epiLVContours = np.zeros([large_num, 3, slices])
    endoRVFWContours = np.zeros([large_num, 3, slices])
    epiRVFWContours = np.zeros([large_num, 3, slices])
    RVSContours = np.zeros([large_num, 3, slices])
    RVInserts = np.zeros([2, 3, slices])

    for i in range(0, slices):
        # Get masks for current slice
        endoLVmask = cleanMask(endoLV[:, :, i])
        epiLVmask = cleanMask(epiLV[:, :, i])
        endoRVmask = cleanMask(endoRV[:, :, i])

        # Get contours from masks. (measure.find_contours returns a list of m x 2 ndarrays; np.stack with axis = 0 converts
        # this list of ndarrays into a single ndarray by concatenating rows).
        tmp_endoLV = getContoursFromMask(endoLVmask)
        tmp_epiLV = getContoursFromMask(epiLVmask)
        tmp_endoRV = getContoursFromMask(endoRVmask)

        # For debug purposes, import MATLAB versions of the above variables. See https://docs.scipy.org/doc/scipy/reference/tutorial/io.html.
        endoLVmaskAll = sio.loadmat(resultsDir + "endoLVmaskAll.mat")["endoLVmaskAll"]
        removedPointsAll = sio.loadmat(resultsDir + "removedPointsAll.mat")["removedPointsAll"]

        tmp_endoLVAll = sio.loadmat(resultsDir + "tmp_endoLVAll.mat")["tmp_endoLVAll"]
        tmp_epiLVAll = sio.loadmat(resultsDir + "tmp_epiLVAll.mat")["tmp_epiLVAll"]

        # Differentiate contours for RVFW (free wall) and RVS (septum)
        [tmp_RVS, ia, ib] = ut.sharedRows(tmp_epiLV, tmp_endoRV)
        tmp_RVFW = tmp_endoRV
        tmp_RVFW = ut.deleteHelper(tmp_RVFW, ib, 0) #In tmpRVFW, delete the row (hence the 0) with index ib.

        # Remove RVS contour points from LV epi contour
        tmp_epiLV = ut.deleteHelper(tmp_epiLV, ia, 0) #In tmp_epiLV, delete the row (hence the 0) with index ia.

        #LV endo
        if tmp_endoLV is not None and tmp_endoLV.size != 0:

            tmp_endoLV = cleanContours(tmp_endoLV, downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            if PLOT and tmp_endoLV is not None and tmp_endoLV.size != 0:
                subplotHelper(title= "LV Endocardium (Python contours on Python mask)", nrows=1, ncols=3, index=1,
                              maskSlice=np.squeeze(endoLV[:, :, i]), contours=tmp_endoLV, color="red", swap=True)

                subplotHelper(title= "LV Endocardium (MATLAB contours on Python mask)", nrows=1, ncols=3, index=1,
                              maskSlice=np.squeeze(endoLV[:, :, i]), contours=np.squeeze(tmp_endoLVAll[:, :, i]),
                              color="green")

                contoursToImageCoords(tmp_endoLV, transform, pix_scale, i, endoLVContours)  # This call writes to "endoLVContours".

        #LV epi
        if tmp_epiLV is not None and tmp_epiLV.size != 0:

            # In this case, we do basically the same thing as above, except with tmp_epiLV, epiLVmask, and epiLVContours instead of
            # tmp_endoLV, endoLVmask, and endoLVContours. A function is not used to generalize the work done by this and
            # the previous case because, since the next case doesn't exactly follow the same format, writing such a function
            # would do more to obfusticate than to simplify).

            tmp_epiLV = cleanContours(tmp_epiLV, downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            if PLOT and tmp_epiLV is not None and tmp_epiLV.size != 0:
                subplotHelper(title = "LV Epicardium (Python contours on Python mask)", nrows = 1, ncols = 3, index = 2,
                              maskSlice = np.squeeze(epiLV[:, :, i]), contours = tmp_epiLV, color = "blue",
                              swap = True)

                subplotHelper(title = "LV Epicardium (MATLAB contours on Python mask)", nrows = 1, ncols = 3, index = 2,
                              maskSlice = np.squeeze(epiLV[:, :, i]), contours = np.squeeze(tmp_epiLVAll[:, :, i]),
                              color = "cyan")

                contoursToImageCoords(tmp_epiLV, transform, pix_scale, i, epiLVContours)  # This call writes to "epiLVContours".

        # RV
        if tmp_RVFW.shape[0] > 6: #~isempty(tmp_RVFW)

            tmp_RVFW = cleanContours(tmp_RVFW, downsample)

            tmpRV_insertIndices = ut.getRVinsertIndices(tmp_RVFW)
            tmp_rvi_indices = ut.getRVinsertIndices(tmp_RVFW)
            tmp_RVS = cleanContours(tmp_RVS, downsample) #possibly switch this and the previous line

            if PLOT and (tmp_RVFW is not None and tmp_RVS is not None) and (tmp_RVFW.size != 0 and tmp_RVS.size != 0):
                ps1 = plotSettings(maskSlice = np.squeeze(endoRV[:, :, i]), contours = tmp_RVFW, color = "green", swap = True)
                ps2 = plotSettings(maskSlice = np.squeeze(endoRV[:, :, i]), contours = tmp_RVS, color = "yellow", swap = True)
                subplotHelperMulti(plotSettingsList = [ps1, ps2], title = "RV Endocardium", nrows = 1, ncols = 3, index = 3)
                pass


# Helper function for converting contours to an image coordinate system. This function writes to "contours".
def contoursToImageCoords(tmp_contours, transform, pix_scale, sliceIndex, contours):
    for i in range(0, tmp_contours.shape[0]):
        pix = np.array([tmp_contours[i, 1], tmp_contours[i, 0], sliceIndex - 1]) * pix_scale
        tmp = transform @ np.append(pix, 1)
        contours[i, :, sliceIndex] = tmp[0:3]

# Helper function used in finalizeContour.
# "contours" is an m1 x 2 ndarray for some m1.
# Returns an m2 x 2 ndarray that is the result of cleaning up "contours".
def cleanContours(contours, downsample):
    if contours.size == 0:
        return np.array([])

    # Remove any points which lie outside of image. Note: simply calling np.nonzero() doesn't do the trick; np.nonzero()
    # returns a tuple, and we want the first element of that tuple.
    indicesToRemove = np.nonzero(contours[:, 0] < 0)[0]
    contours = ut.deleteHelper(contours, indicesToRemove, 1)  # The 1 corresponds to deleting a column.

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

def subplotHelperMulti(plotSettingsList, title, nrows, ncols, index):
    for ps in plotSettingsList:
        subplotHelper(title, nrows, ncols, index, ps.maskSlice, ps.contours, ps.color, ps.swap)

# Helper function for creating subplots.
def subplotHelper(title, nrows, ncols, index, maskSlice, contours, color, swap = False):
   # Adjust basic plot features.
    plt.subplot(nrows, ncols, index)  # Going against Python conventions, indexing for subplots starts at 1.
    plt.title(title)
    plt.imshow(maskSlice) # This is the "background image" on top of which the contours will be displayed.
    plt.axis("equal") # Equal aspect ratio.
    plt.set_cmap("gray") # Colormap.

    # Rotate and flip the contours if necessary.
    xx = contours[:, 0]
    yy = contours[:, 1]
    if swap: #swap xx and yy
        tmp = xx
        xx = yy
        yy = tmp

   # Scatterplot. "s" is the size of the points in the scatterplot.
    plt.scatter(x = xx, y = yy, s = .5, c = color)

    plt.show(block = False)

# Helper function to call np.stack() when appropriate (i.e. when the list argument to np.stack() is nonempty).
def getContoursFromMask(mask):
    lst = measure.find_contours(mask, 0) #list is a list of m x 2 ndarrays.

    # This "else" in the below is the reason this function is necessary; np.stack does not accept an empty list.
    return np.vstack(lst) if not len(lst) == 0 else np.array([])

# Helper function for removing irregularities from masks.
def cleanMask(mask):
    #Remove irregularites
    morphology.remove_small_objects(np.squeeze(mask), min_size = 50, connectivity = 2) # Might have to come back and see if connectivity = 2 was the right choice

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    return morphology.reconstruction(seed, mask, method='erosion')

# This function writes to "contours", which is a m1 x n1 x p1 ndarray for some m1, n1, p1. In practice,
# "contours" is either "endoLVContours" or "epiLVContours".
#
# "mask" is an m2 x n2 x p2 ndarray for some m2, n2, p2. In practice, it is "endoLV" or "epiLV", and is something like
# a 256 x 256 x 15 ndarray.
# "tmp_contours" is an m3 x 2 ndarray for some m3. In practice, it is "tmp_endoLV" or "tmp_epiLV".
# "more_contours" is an m4 x 2 ndarray for some m4. In practice, it is "tmp_endoLVAll" or "tmp_epiLVAll".
# "sliceIndex" will be the index "i" inside a for-loop over the third dimension (the p-dimension) of "mask".
# "transform", "pix_scale", "downsample" are meant to be "transform", "pix_scale", and "downsample" from outside this function.
# If and only if "PLOT" is True, a plot of the contours will be generated. "plot_title" is then the title of this plot.
# def finalizeContour(mask, tmp_contours, more_contours, sliceIndex, transform, pix_scale, downsample, PLOT, plot_title, contours):
#     tmp_contours = cleanContours(tmp_contours, downsample)
#
#     # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
#     if PLOT and tmp_contours is not None and tmp_contours.size != 0:
#         subplotHelper(title = plot_title + " (Python contours on Python mask)", nrows=1, ncols=3, index=1,
#                       maskSlice=np.squeeze(mask[:, :, sliceIndex]), contours=tmp_contours, color="red", swap=True)
#
#         subplotHelper(title = plot_title + " (MATLAB contours on Python mask)", nrows=1, ncols=3, index=2,
#                       maskSlice=np.squeeze(mask[:, :, sliceIndex]), contours=np.squeeze(more_contours[:, :, sliceIndex]),
#                       color="green")
#
#         contoursToImageCoords(tmp_contours, transform, pix_scale, sliceIndex, contours) # This call writes to "contours".
