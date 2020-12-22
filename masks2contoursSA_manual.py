import numpy as np
import nibabel as nib
from skimage import morphology
from skimage import measure
import matplotlib.axes as ax
import matplotlib.pyplot as plt
import scipy.io as sio

#TO-DO: run Renee's new examples and compare contours
#TO-DO: hardcode in points to remove to make testing easier
#TO-DO: are axes getting swapped somehow?

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
        [tmp_RVS, ia, ib] = sharedRows(tmp_epiLV, tmp_endoRV)
        tmp_RVFW = tmp_endoRV
        tmp_RVFW = deleteHelper(tmp_RVFW, ib, 0) #In tmpRVFW, delete the row (hence the 0) with index ib.

        # Remove RVS contour points from LV epi contour
        tmp_epiLV = deleteHelper(tmp_epiLV, ia, 0) #In tmp_epiLV, delete the row (hence the 0) with index ia.

        #LV endo
        if tmp_endoLV is not None and tmp_endoLV.size != 0:
            #(This is a debug comment; delete later). should be nonempty for slices 3 through 12 (at least)

            print("LV endo") #debug

            finalizeContour(mask=endoLV, tmp_contours=tmp_endoLV, tmp_contours_debug=tmp_endoLVAll, sliceIndex=i,
                            transform=transform, pix_scale=pix_scale, downsample=downsample, PLOT=PLOT,
                            plot_title = "LV Endocardium", contours=endoLVContours)

        if tmp_epiLV is not None and tmp_epiLV.size != 0:

            print("LV epi") #debug

            finalizeContour(mask=epiLV, tmp_contours=tmp_epiLV, tmp_contours_debug=tmp_epiLVAll, sliceIndex=i,
                            transform=transform, pix_scale=pix_scale, downsample=downsample, PLOT=PLOT,
                            plot_title = "LV Epicardium", contours=epiLVContours)

        #TO DO: Condense the "LV endo" and "LV epi" sections from the MATLAB code into a single function. Make sure to pass contours (coorresp. to endoLVContours or epiLVContours) by reference.
        #TO DO: After previous step completed, generalize the function further so that it handles the "RV" section from the MATLAB code.

        # For debug purposes
        # plt.imshow(endoLVmask, interpolation = "none")
        # plt.show()


#
# Writes to contours
#
def finalizeContour(mask, tmp_contours, tmp_contours_debug, sliceIndex, transform, pix_scale, downsample, PLOT, plot_title, contours):
    # Remove any points which lie outside of image.
    indicesToRemove = np.nonzero(tmp_contours[:, 0] < 0)[0]  # Simply calling np.nonzero() doesn't do the trick;
    # np.nonzero() returns a tuple, and we want the first element of that tuple.
    tmp_endoLV = deleteHelper(tmp_contours, indicesToRemove, 1)  # The 1 corresponds to deleting a column.

    # Downsample.
    tmp_endoLV = tmp_endoLV[0:(tmp_endoLV.size - 1):downsample, :]

    # Remove points in the contour that are too far from the majority of the contour. (This probably happens due to holes in segmentation).
    tmp_endoLV = removeFarPoints(tmp_endoLV)

    # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
    if PLOT and tmp_endoLV is not None and tmp_endoLV.size != 0:
        print("Plotting endoLV contours (tmp_endoLV).")
        subplotHelper(title = plot_title + " (Python contours on Python mask)", nrows=1, ncols=3, index=1,
                      maskSlice=np.squeeze(mask[:, :, sliceIndex]), contours=tmp_endoLV, color="red", swap=True)

        subplotHelper(title = plot_title + " (MATLAB contours on Python mask)", nrows=1, ncols=3, index=2,
                      maskSlice=np.squeeze(mask[:, :, sliceIndex]), contours=np.squeeze(tmp_contours_debug[:, :, sliceIndex]),
                      color="green")

        # Convert contours to image coordinate system.
        for i in range(0, tmp_endoLV.shape[0] - 1):
            pix = np.array([tmp_endoLV[i, 1], tmp_endoLV[i, 0], sliceIndex - 1]) * pix_scale
            tmp = transform @ np.append(pix, 1)
            contours[i, :, sliceIndex] = tmp[0:3]

#Helper function for creating subplots.
def subplotHelper(title, nrows, ncols, index, maskSlice, contours, color, swap = False):
   #Adjust basic plot features.
    plt.subplot(nrows, ncols, index)  # Going against Python conventions, indexing for subplots starts at 1.
    plt.title(title)
    plt.imshow(maskSlice) #this is the "background image" on top of which the contours will be displayed
    plt.axis("equal") #equal aspect ratio
    plt.set_cmap("gray") #colormap

    #Set up the scatterplot.
    xx = contours[:, 0]
    yy = contours[:, 1]
    if swap: #swap xx and yy
        tmp = xx
        xx = yy
        yy = tmp
    plt.scatter(x = xx, y = yy, c = color)  # Swapped the parameters for x and y.

    plt.show()

#Helper function to call np.stack() when appropriate (i.e. when the list argument to np.stack() is nonempty).
def getContoursFromMask(mask):
    lst = measure.find_contours(mask, 0) #list is a list of m x 2 ndarrays.

    # This "else" in the below is the reason this function is necessary; np.stack does not accept an empty list.
    return np.vstack(lst) if not len(lst) == 0 else np.array([])

def cleanMask(mask):
    #Remove irregularites
    morphology.remove_small_objects(np.squeeze(mask), min_size = 50, connectivity = 2) # Might have to come back and see if connectivity = 2 was the right choice

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = mask.max()
    return morphology.reconstruction(seed, mask, method='erosion')

# "oldPoints" is a m1 x 2 ndarray for some m1.
#
# Returns a m2 x 2 ndarray, for some m2, that is the result of removing points that are far from the contour of
# interest from "oldPoints".
def removeFarPoints(oldPoints):
    # Record the distances between consecutive points (wrap around from the last point to the first point) in oldPoints.
    numPts = oldPoints.shape[0]
    distances = []
    for i in range(0, numPts - 1):
        if i == numPts - 1: #If on last point, compare last point with first point
            distances.append(np.linalg.norm(oldPoints[i, :] - oldPoints[0, :]))
        else:
            distances.append(np.linalg.norm(oldPoints[i + 1, :] - oldPoints[i, :]))

    distances = np.array(distances)

    print("distances = {}".format(distances))

    # Find points that are far away from other points.
    far_indices = np.nonzero(distances > (np.mean(distances) + 2 * np.std(distances)))[0] #Simply calling np.nonzero() doesn't do the trick;
    # np.nonzero() returns a tuple, and we want the first element of that tuple.

    # The point must be far from its two nearest neighbours, therefore, if there are extraneous points, the length of
    # 'far_indices' must be even. If the length of 'far_indices' is odd, the contour is open (but there still may be
    # extraneous points).

    print("far_indices = {}".format(far_indices))

    # Find these extraneous points and remove them.
    if np.size(far_indices) > 1:
        indicesToRemove = []
        for i in range(0, np.size(far_indices) - 2):
            if (far_indices[i] == 0) and (far_indices[-1] == np.size(oldPoints) - 1):
                indicesToRemove.append(far_indices[i])
            elif far_indices[i + 1] - far_indices[i] == 0:
                indicesToRemove.append(far_indices[i + 1])

        print("indices to remove = {}".format(indicesToRemove))

        print("points to remove = {}".format(oldPoints[indicesToRemove]))

        # Remove the extraneous points.
        return deleteHelper(oldPoints, indicesToRemove)

#Helper function for deleting parts of ndarrays that allows for empty indices.
#
# arr may either be a list or an ndarray.
def deleteHelper(arr, indices, axis = 0):
    #np.delete() does not handle the case in which arr is an empty list or an empty ndarray.
    if (type(arr) is np.ndarray and arr.size == 0) or (type(arr) is list and len(list) == 0):
        return arr
    elif type(indices) is np.ndarray and indices.size == 0: # np.delete() does not work as expected if arr is an empty ndarray. This case fixes that. (If indices is an empty list, np.delete() works as expected).
        return arr
    else:
        return np.delete(arr, indices, axis)

#
# arr1 and arr2 must be m x n ndarrays, for some m and n. (So they can't be m x n x s numpy arrays).
#
# This function returns the list [_sharedRows, sharedIndicesArr1, sharedIndicesArr2].
# _sharedRows is a matrix whose rows are those that are shared by arr1 and arr2.
# sharedIndicesArr1 is a list of the row-indices in arr1 whose rows appear (not necessarily with the same indices) in
# arr2.
# sharedIndicesArr2 is a similar list of row-indices that pertains to arr2.
def sharedRows(arr1, arr2):
    if arr1.size == 0 or arr2.size == 0: #If either array is empty, return a list containing three empty ndarrays.
        return[np.array([]), np.array([]), np.array([])]

    # First, just get the indices of the shared rows.
    sharedIndicesArr1 = []
    sharedIndicesArr2 = []
    jRange = list(range(0, arr2.shape[1])) # Needs to be a list so we can use remove()
    for i in range(0, arr1.shape[0]):
        for j in jRange:
            if np.all(arr1[i, :] == arr2[j, :]): #If (ith row in arr1) == (ith row in arr2)
                sharedIndicesArr1.append(i)
                sharedIndicesArr2.append(j)
                jRange.remove(j) # After we visit a row in arr2, we don't need to visit it again.

    # Use these indices to build the matrix of shared rows.
    _sharedRows = [arr1[i] for i in sharedIndicesArr1]

    # Convert everything to ndarrays.
    _sharedRows = np.array(_sharedRows)
    sharedIndicesArr1 = np.array(sharedIndicesArr1)
    sharedIndicesArr2 = np.array(sharedIndicesArr2)

    return [_sharedRows, sharedIndicesArr1, sharedIndicesArr2]