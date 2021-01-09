import numpy as np
import nibabel as nib
from skimage import morphology
import masks2contoursSA_manual as m2cSA
import masks2contours_util as ut
import matplotlib.pyplot as plt

def masks2contoursLA_manual(LA_names, LA_segs, resultsDir, frameNum, config):
    # Initialize variables.
    ub = config.upperBdNumContourPts
    numSlices = len(LA_names)
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
        #Get name of current LA view
        # !! This is unique to LA !!
        tmp = LA_segs[i].split("\\")
        tmpLA = tmp[-1].split("_")

        # Load long axis segmentations and header info
        _seg = nib.load(LA_segs[i])
        seg = _seg.get_fdata()
        info = _seg.header

        if seg.ndim > 3:  # if segmentation includes all time points
            seg = seg[:, :, 0, frameNum].squeeze()

        # Haven't tested if seg is loaded correctly yet.

        # Obtain the 4x4 homogeneous affine matrix
        transform = _seg.affine  # In the MATLAB script, we transposed the transform matrix at this step. We do not need to do this here due to how nibabel works.
        transform[:2, :] = transform[:2, :] * -1  # This edit has to do with RAS system in Nifti files

        # Initialize pix_scale. In the MATLAB script, pix_scale was a column vector. Here, it will be a row vector.
        pixdim = info.structarr["pixdim"][1:3 + 1]
        if np.isclose(np.linalg.norm(transform[:, 0]), 1):
            pix_scale = pixdim  # Here we are manually going into the header file's data. This is somewhat dangerous- maybe it can be done a better way?
        else:
            pix_scale = np.array([1, 1, 1])

        # Initialize some more things.
        pixSpacing = pixdim[0]

        # Separate out LV endo, LV epi, and RV endo segmentations.
        endoLV = seg == 1
        epiLV = (seg <= 2) * (seg > 0)
        endoRV = seg == 3

        # Get masks for current slice
        # !! Unique to LA:
        # - cleanMask is called on endoLV rather than on endoLV[:, :, i], etc.
        # - np.squeeze is called after cleanMask !!
        endoLVmask = np.squeeze(m2cSA.cleanMask(endoLV, irregMaxSize = 20))
        epiLVmask = np.squeeze(m2cSA.cleanMask(epiLV, irregMaxSize = 20))
        endoRVmask = np.squeeze(m2cSA.cleanMask(endoRV, irregMaxSize = 20))

        # Get contours from masks.
        print(endoLVmask.shape)
        tmp_endoLV = m2cSA.getContoursFromMask(endoLVmask)
        tmp_epiLV = m2cSA.getContoursFromMask(epiLVmask)
        tmp_endoRV = m2cSA.getContoursFromMask(endoRVmask)

        # Differentiate contours for RVFW (free wall) and RVS (septum)
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

        # everything same except for imagesc(endoLV) instead of imagesc(squeeze(endoLV(:, :, i)))
        # and pix = [tmp_endoLV(j,2); tmp_endoLV(j,1); 0] .* pix_scale;
        # instead of pix = [tmp_endoLV(j,2); tmp_endoLV(j,1); i - 1] .* pix_scale;

        # LV endo
        if not LVEndoIsEmpty:

            tmp_endoLV = m2cSA.cleanContours(tmp_endoLV, config.downsample)

            # If a plot is desired and contours exist, plot the mask and contour. (Contours might not exist, i.e. tmp_endoLV might be None, due to the recent updates).
            LVEndoIsEmptyAfterCleaning = tmp_endoLV is None or tmp_endoLV.size == 0
            if config.PLOT and not LVEndoIsEmptyAfterCleaning:
                # !! unique to LA !!:
                # "maskSlice = np.squeeze(endoLV)" instead of "maskSlice = np.squeeze(endoLV[:, :, i]) #
                # swap = True instead of swap = not LOAD_MATLAB_VARS
                m2cSA.subplotHelper(axs[0], title = "LV Endocardium", maskSlice = endoLV,
                              contours = tmp_endoLV, color = "red",
                              swap = True)  # Python contours on Python mask

                # subplotHelper(axs[0], title = "LV Endocardium", maskSlice = np.squeeze(endoLV[:, :, i]),
                # contours = np.squeeze(tmp_endoLVAll[:, :, i]), color = "green") # MATLAB contours on Python mask

                m2cSA.contoursToImageCoords(tmp_endoLV, transform, pix_scale, i,
                                      endoLVContours, "LA")  # This call writes to "endoLVContours".

        # Show the figure for .5 seconds if config.PLOT is True.
        if config.PLOT and not (LVEndoIsEmpty or LVEpiIsEmpty or RVEndoIsEmpty):
            fig.show()
            while True:
                #if plt.waitforbuttonpress(): break
                if plt.waitforbuttonpress(timeout = .5) is None: break