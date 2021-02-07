import alphashape
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

    # Loop over slices and get contours one slice at a time.
    for i in range(0, numSlices):
        inputsList = (endoLV[:, :, i], epiLV[:, :, i], endoRV[:, :, i], transform, pixSpacing) #Note the last 3 args are the same every iteration.
        outputsList = (endoLVContours, epiLVContours, RVSContours, endoRVFWContours, epiRVFWContours, RVInserts)
        figaxs = None
        if config.PLOT:  # figaxs is possibly None (but this is fine because it is only used when config.PLOT == True)
            figaxs = (fig, axs)
        slice2Contours(inputsList, outputsList, config, figaxs, i, "SA")

    # Now, calculate weights for RV insertion points.

    # Loop through anterior (1) and posterior (2) sides for fitting a line.
    RVInsertsWeights = np.zeros((2, numSlices)) # This will store errors.
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

    # Return a 2-tuple of dictionaries.
    return ({"endoLV" : endoLVContours,
            "epiLV" : epiLVContours,
            "endoRVFW" : endoRVFWContours,
            "epiRVFW" : epiRVFWContours,
            "RVSept" : RVSContours } ,
            {"RVInserts" : RVInserts, "RVInsertsWeights" : RVInsertsWeights})

def masks2ContoursLA(LA_segs, resultsDir, frameNum, numSlices, config):
    # Precompute (more accurately, "pre-read") endoLV, epiLV, endoRV for each slice.
    # Also precompute transform and pix_spacing for each slice.
    (endoLVList, epiLVList, endoRVList, transformList, pixSpacingList) = ([], [], [], [], [])
    for i in range(0, numSlices):
        (seg, transform, pixSpacing) = readFromNIFTI(LA_segs[i], frameNum)
        endoLVList.append(np.squeeze(seg == 1))
        epiLVList.append(np.squeeze((seg <= 2) * (seg > 0)))
        endoRVList.append(np.squeeze(seg == 3))
        transformList.append(transform)
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

    # Loop over slices and get contours one slice at a time.
    for i in range(0, numSlices):
        # Get name of current LA view (this seems unnecessary but I've ported it over anyway).
        tmp = LA_segs[i].split("\\")
        tmpLA = tmp[-1].split("_")

        inputsList = (endoLVList[i], epiLVList[i], endoRVList[i], transformList[i], pixSpacingList[i])
        outputsList = (endoLVContours, epiLVContours, RVSContours, endoRVFWContours, epiRVFWContours)
        figaxs = None
        if config.PLOT: #figaxs is possibly None (but this is fine because it is only used when config.PLOT == True)
            figaxs = (fig, axs)
        slice2Contours(inputsList, outputsList, config, figaxs, i, "LA")

    # Return a dictionary.
    return {"endoLV": endoLVContours,
            "epiLV": epiLVContours,
            "endoRVFW": endoRVFWContours,
            "epiRVFW": epiRVFWContours,
            "RVSept": RVSContours}

def slice2Contours(inputsList, outputsList, config, figaxs, sliceIndex, SA_LA):
    # Check validity of SA_LA.
    SA_LA = SA_LA.lower()
    if not(SA_LA == "sa" or SA_LA == "la"):
        raise ValueError("SA_LA must either be \"SA\" or \"LA\".")

    # Unpack tuples passed in as input.
    (endoLV, epiLV, endoRV, transform, pixSpacing) = inputsList

    if SA_LA == "sa":
        (endoLVContours, epiLVContours, RVSContours, endoRVFWContours, epiRVFWContours, RVInserts) = outputsList
    else: # SA_LA == "la"
        (endoLVContours, epiLVContours, RVSContours, endoRVFWContours, epiRVFWContours) = outputsList

    if config.PLOT:
        (fig, axs) = figaxs

    # Get contours for each slice in the different regions of the heart.
    LVendoSContours = getContoursFromMask(endoLV, irregMaxSize = 20)
    LVepiSContours = getContoursFromMask(epiLV, irregMaxSize = 20)
    RVendoSContours = getContoursFromMask(endoRV, irregMaxSize = 20)

    ############ DEBUG ##########
    # Load MATLAB variables.
    if config.LOAD_MATLAB_VARS:
        import scipy.io as sio
        resultsDir = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\out"

        suffix = "LA" if SA_LA == "la" else ""
        vars = ["tmp_endoLV", "tmp_epiLV", "tmp_endoRV"]
        result = []
        for x in vars:
            fldr = resultsDir + "\\" + x + suffix + "_slices\\"
            file = fldr + x + suffix + "_slice_" + str(sliceIndex + 1) + ".mat"
            result.append(sio.loadmat(file)[x])

        [LVendoMAT, LVepiMAT, RVendoMAT] = result

        # Replace the Python vars with the MATLAB ones.
        [LVendoSContours, LVepiSContours, RVendoSContours] = [LVendoMAT, LVepiMAT, RVendoMAT] # redundant, but gets point across best

    # For some reason, it's necessary to swap the axes when nibabel is used to load the nifti files.
    if not config.LOAD_MATLAB_VARS:
        LVEndoIsEmpty = LVendoSContours is None or np.max(LVendoSContours.shape) <= 2
        LVEpiIsEmpty = LVepiSContours is None or np.max(LVepiSContours.shape) <= 2
        RVEndoIsEmpty = RVendoSContours is None or RVendoSContours.size == 0
        if not LVEndoIsEmpty:
            LVendoSContours[:, [0, 1]] = LVendoSContours[:, [1, 0]]
        if not LVEpiIsEmpty:
            LVepiSContours[:, [0, 1]] = LVepiSContours[:, [1, 0]]
        if not RVEndoIsEmpty:
            RVendoSContours[:, [0, 1]] = RVendoSContours[:, [1, 0]]

    # Differentiate contours for RVFW (free wall) and RVS (septum).
    [RVSeptSContours, ia, ib] = ut.sharedRows(LVepiSContours, RVendoSContours)
    RVFWSContours = RVendoSContours
    RVFWSContours = ut.deleteHelper(RVFWSContours, ib, axis = 0)  # In tmpRVFW, delete the rows with index ib.

    # Remove RVS contour points from LV epi contour
    LVepiSContours = ut.deleteHelper(LVepiSContours, ia, axis = 0)  # In LVepiSContours, delete the rows with index ia.

    # If doing long axis, remove line segments at base which are common to endoLV and epiLV.
    if SA_LA == "la":
        [wont_use_this_var, ia, ib] = ut.sharedRows(LVendoSContours, LVepiSContours)
        LVendoSContours = ut.deleteHelper(LVendoSContours, ia, axis = 0)
        LVepiSContours = ut.deleteHelper(LVepiSContours, ib, axis = 0)

    # These variables are for readability.
    LVEndoIsEmpty = LVendoSContours is None or np.max(LVendoSContours.shape) <= 2
    LVEpiIsEmpty = LVepiSContours is None or np.max(LVepiSContours.shape) <= 2
    RVEndoIsEmpty = RVFWSContours is None or RVFWSContours.size == 0

    # LV endo
    if not LVEndoIsEmpty:

        LVendoSContours = cleanContours(LVendoSContours, config.downsample)

        # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. LVendoSContours might be None, due to the recent updates).
        LVEndoIsEmptyAfterCleaning = LVendoSContours is None or LVendoSContours.size == 0
        if config.PLOT and not LVEndoIsEmptyAfterCleaning:
            subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLV),
                          contours = LVendoSContours, color = "red")
        contoursToImageCoords(LVendoSContours, transform, sliceIndex, endoLVContours, SA_LA)  # This call writes to "endoLVContours".

    # LV epi
    if not LVEpiIsEmpty:

        # In this case, we do basically the same thing as above, except with LVepiSContours, epiLVmask, and epiLVContours instead of
        # LVendoSContours, endoLVmask, and endoLVContours. A function is not used to generalize the work done by this and
        # the previous case because, since the next case doesn't exactly follow the same format, writing such a function
        # would do more to obfusticate than to simplify).

        LVepiSContours = cleanContours(LVepiSContours, config.downsample)

        # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. LVendoSContours might be None, due to the recent updates).
        LVEpiIsEmptyAfterCleaning = LVendoSContours is None or LVendoSContours.size == 0
        if config.PLOT and not LVEpiIsEmptyAfterCleaning:
            subplotHelper(axs[1], title = "LV Epicardium", maskSlice = np.squeeze(epiLV),
                          contours = LVepiSContours, color = "blue")
        contoursToImageCoords(LVepiSContours, transform, sliceIndex, epiLVContours, SA_LA)  # This call writes to "epiLVContours".

    # RV
    if not RVEndoIsEmpty:
        RVFWSContours = cleanContours(RVFWSContours, config.downsample)
        RVSeptSContours = cleanContours(RVSeptSContours, config.downsample)

        # If doing short axis, get the indices corresponding to insertion points in the RV.
        if SA_LA == "sa":
            tmpRV_insertIndices = ut.getRVinsertIndices(RVFWSContours)

        RVEndoIsEmptyAfterCleaning = (RVFWSContours is None or RVSeptSContours is None) or (RVFWSContours.size == 0 or RVSeptSContours.size == 0)

        # If doing long axis, remove basal line segment in RVFW LA contour.
        if SA_LA == "la" and not RVEndoIsEmptyAfterCleaning:
            figI, axI = plt.subplots() # I for "inspection"
            title = "Select points you would like to remove. Click and drag to lasso select.\n The zoom tool may also be helpful."
            pts = subplotHelper(axI, title = title, maskSlice = np.squeeze(endoRV), contours = RVFWSContours, color = "green", size = 20)

            # Create a lasso selector. It automatically is able to be used after plt.show().
            lassoSelector = SelectFromCollection(figI, axI, pts)
            plt.show() # Important to use plt.show() instead of figI.show() so that the event loop runs. See https://github.com/matplotlib/matplotlib/issues/13101#issuecomment-452032924

            # Remove the points that were selected from the contour.
            RVSeptSContours = ut.deleteHelper(RVFWSContours, lassoSelector.ind, axis = 0)

            # After the user has pressed "Enter", control will be returned to this point.

            if config.PLOT:
                # The fig.show() call at the very end of this function will not work unless we steal some "dummy" figure's figure manager
                # and give it to fig.
                #
                # This is because plt.close("all"), which is called by the callback function that responds
                # when "Enter" is pressed, destroys all figure managers owned by figures that are shown. (And we
                # unfortunately do have to use plt.close("all") in order to get the matplotlib event loop to exit).
                #
                # The following code was taken from https://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed.
                dummy = plt.figure()
                new_manager = dummy.canvas.manager
                new_manager.canvas.figure = fig
                fig.set_canvas(new_manager.canvas)

        if config.PLOT and not RVEndoIsEmptyAfterCleaning:
            ps1 = PlotSettings(maskSlice = np.squeeze(endoRV), contours = RVSeptSContours, color ="yellow")
            ps2 = PlotSettings(maskSlice = np.squeeze(endoRV), contours = RVFWSContours, color ="green")
            subplotHelperMulti(axs[2], plotSettingsList = [ps1, ps2], title = "RV Endocardium")

        contoursToImageCoords(RVSeptSContours, transform, sliceIndex, RVSContours, SA_LA)
        contoursToImageCoords(RVFWSContours, transform, sliceIndex, endoRVFWContours, SA_LA)

        # If doing short axis, save RV insert coordinates.
        if SA_LA == "sa" and (tmpRV_insertIndices is not None and tmpRV_insertIndices.size != 0):
            RVInserts[:, :, sliceIndex] = endoRVFWContours[tmpRV_insertIndices[0:1 + 1], :, sliceIndex]

        # Calculate RV epicardial wall by applying a thickness to RV endocardium.
        if SA_LA == "sa":
            RVEndoNormals = ut.lineNormals2D(RVFWSContours)
            RVepiSlice = RVFWSContours - np.ceil(config.rvWallThickness / pixSpacing) * RVEndoNormals \
                if RVFWSContours.size != 0 and RVEndoNormals.size != 0 else np.array([])
            contoursToImageCoords(RVepiSlice, transform, sliceIndex, epiRVFWContours, "SA")
        else:
            # Double check that normal is pointing towards epicardium by checking to see if epi points are inside the alpha shape
            # created by the endocardial points.
            RVEndoNormals = ut.lineNormals2D(RVFWSContours)
            RVepiSlice = RVFWSContours + RVEndoNormals * config.rvWallThickness / pixSpacing
            divByZeroIndices = np.any(np.isinf(RVepiSlice), axis = 1)
            RVepiSlice = ut.deleteHelper(RVepiSlice, divByZeroIndices, axis = 0)

            tmp_RV = np.vstack((RVFWSContours, RVSeptSContours))
            alpha = alphashape.optimizealpha(tmp_RV)
            shape = alphashape.alphashape(tmp_RV, alpha * 2)

            # FINISH THIS else STATEMENT #

            # https://gis.stackexchange.com/questions/208546/check-if-a-point-falls-within-a-multipolygon-with-python

            # countIN = inShape(a, RVepiSlice(:,1), RVepiSlice(:,2));
            # if sum(countIN)/length(countIN) > 0.5 % If more than 50% of epi points are inside of the alpha shape
            #    RVepiSlice = RVFWSContours - N*(rv_wall/pixSpacing); % Reverse the normal direction

            contoursToImageCoords(RVepiSlice, transform, sliceIndex, epiRVFWContours, "LA")

    # Show the figure for .5 seconds if config.PLOT is True.
    if config.PLOT and not (LVEndoIsEmpty or LVEpiIsEmpty or RVEndoIsEmpty):
        fig.show()
        while True:
            if plt.waitforbuttonpress(): break
            #if plt.waitforbuttonpress(timeout = .5) is None: break

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