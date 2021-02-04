from glob import glob
from typing import NamedTuple

import nibabel as nib
import numpy as np

import masks2ContoursMainUtil as mut
import masks2ContoursUtil as ut
from masks2Contours import masks2ContoursSA, masks2ContoursLA


def main():
    np.set_printoptions(suppress = True) # For debugging: don't use scientific notation when printing out ndarrays.

    # Orientation for viewing plots
    az = 214.5268
    el = -56.0884

    # Configure some settings.
    config = Config(rvWallThickness = 3, downsample = 3, upperBdNumContourPts = 200, PLOT = True, LOAD_MATLAB_VARS = True)
    frameNum = 1 # Index of frame to be segmented.

    # Get filepaths ready.
    fldr = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\"
    resultsDir = fldr + "out\\"

    # These are filepaths for the SA.
    imgName = fldr + "CINE_SAX.nii"
    segName = fldr + "CINE_SAX_" + str(frameNum) + ".nii"

    # These are lists of filepaths for the LA.
    LA_segs = glob(fldr + "LAX_[0-9]ch_1.nii")
    LA_names = glob(fldr + "RReg_LAX_[0-9]ch_to_Aligned_SA.nii")

    # The following two functions, masks2ContoursSA() and masks2ContoursLA(), produce contour points from masks for the
    # short and long axis image files, respectively. If PLOT == True, these contour points are displayed slice by slice
    # as they are computed.
    #
    # SAContours and LAContours are dicts whose keys are strings such as "LVendo" or "RVSept" and whose values are
    # m x 2 ndarrays containing the contour points. SAinserts is a dict with the two keys "RVInserts" and "RVInsertsWeights".
    (SAContours, SAinserts) = masks2ContoursSA(segName, frameNum, config)
    LAContours = masks2ContoursLA(LA_segs, resultsDir, frameNum, numSlices = len(LA_names), config = config)

    # Get valve points for mitral valve (mv), tricuspid valve (tv), aortic valve (av), and pulmonary valve (pv).
    # Note: the valve points computed by this Python script are slightly different than those produced by MATLAB because
    # the interpolation function used in this code, np.interp(), uses a different interplation method than the MATLAB interp1().
    numFrames = nib.load(imgName).get_fdata().shape[3]  # all good
    (mv, tv, av, pv) = mut.manuallyCompileValvePoints(fldr, numFrames, frameNum)

    # Remove rows that are all zero from mv, tv, av, pv.
    mv = np.reshape(mv, (-1, 3)) # Reshape mv to have shape m x 3 for some m (the -1 indicates an unspecified value).
    mv = ut.removeZerorows(mv)
    tv = ut.removeZerorows(tv)
    av = ut.removeZerorows(av)
    pv = ut.removeZerorows(pv)

    # Omit short axis slices without any contours. Initialize includedSlices to hold the slices we will soon iterate over.
    SAendoLVContours = SAContours["endoLV"]
    slicesToKeep = np.squeeze(SAendoLVContours[0, 0, :])
    slicesToOmit = np.logical_not(slicesToKeep).astype(int) # is 1 wherever slicesToKeep was 0.

    numSASlices = SAendoLVContours.shape[2]
    includedSlices = np.linspace(1, numSASlices, numSASlices)
    includedSlices = ut.deleteHelper(includedSlices, slicesToOmit, axis = 0).astype(int)
    includedSlices = includedSlices - 1 # convert to Python indexing

    # Calculate apex using "method 2" from the MATLAB script.
    LAepiLVContours = LAContours["epiLV"]
    epiPts1 = np.squeeze(LAepiLVContours[:, :, 0])
    epiPts2 = np.squeeze(LAepiLVContours[:, :, 2])
    apex = mut.calcApex(epiPts1, epiPts2)

    # =========================
    # Plot the results.
    # =========================
    import matplotlib.pyplot as plt
    # no function from this import is used.

    # Short axis contours.
    fig, ax = plt.subplots(1, 1, subplot_kw = {"projection" : "3d"})
    for i in includedSlices:
        endoLV = prepareContour("endoLV", SAContours, i)
        epiLV = prepareContour("epiLV", SAContours, i)
        endoRVFW = prepareContour("endoRVFW", SAContours, i)
        epiRVFW = prepareContour("epiRVFW", SAContours, i)
        RVSept = prepareContour("RVSept", SAContours, i)
        RVInserts = prepareContour("RVInserts", SAinserts, i)

        h0 = ax.scatter(endoLV[:, 0], endoLV[:, 1], endoLV[:, 2], marker = ".", color = "green")
        h1 = ax.scatter(epiLV[:, 0], epiLV[:, 1], epiLV[:, 2], marker = ".", color = "blue")
        h2 = ax.scatter(endoRVFW[:, 0], endoRVFW[:, 1], endoRVFW[:, 2], marker = ".", color = "red")
        h3 = ax.scatter(epiRVFW[:, 0], epiRVFW[:, 1], epiRVFW[:, 2], marker = ".", color = "yellow")
        h4 = ax.scatter(RVSept[:, 0], RVSept[:, 1], RVSept[:, 2], marker = ".", color = "blue")
        h5 = ax.scatter(RVInserts[:, 0], RVInserts[:, 1], RVInserts[:, 2], s = 50, color = "red")

    # Valve points.
    h6 = ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s = 50, color = "black")
    h7 = ax.scatter(av[:, 0], av[:, 1], av[:, 2], s = 50, color = "cyan")
    h8 = ax.scatter(tv[:, 0], tv[:, 1], tv[:, 2], s = 50, color = "magenta")

    # Apex.
    h9 = ax.scatter(apex[0], apex[1], apex[2], color = "black")

    # Long axis contours.
    numLASlices = LAContours["endoLV"].shape[2]
    for i in range(numLASlices):
        endoLV = prepareContour("endoLV", LAContours, i)
        epiLV = prepareContour("epiLV", LAContours, i)
        endoRVFW = prepareContour("endoRVFW", LAContours, i)
        epiRVFW = prepareContour("epiRVFW", LAContours, i)
        RVSept = prepareContour("RVSept", LAContours, i)

        ax.scatter(endoLV[:, 0], endoLV[:, 1], endoLV[:, 2], marker = ".", color = "green")
        ax.scatter(epiLV[:, 0], epiLV[:, 1], epiLV[:, 2], marker = ".", color = "blue")
        ax.scatter(endoRVFW[:, 0], endoRVFW[:, 1], endoRVFW[:, 2], marker = ".", color = "red")
        ax.scatter(epiRVFW[:, 0], epiRVFW[:, 1], epiRVFW[:, 2], marker = ".", color = "yellow")
        ax.scatter(RVSept[:, 0], RVSept[:, 1], RVSept[:, 2], marker = ".", color = "blue")

    ax.view_init(elev = el, azim= az)
    ax.legend((h0, h1, h2, h3, h4, h5, h6, h7, h8, h9),
              ("LV endo", "Epi", "RVSept", "RVFW endo", "RVFW epi", "RV inserts",
              "Mitral valve", "Aortic valve", "Tricuspid valve", "Apex"))
    plt.show()  # Must use plt.show() instead of fig.show(). Might have something to do with https://github.com/matplotlib/matplotlib/issues/13101#issuecomment-452032924

def prepareContour(varName, contoursDict, sliceIndex):
    result = contoursDict[varName]
    result = np.squeeze(result[:, :, sliceIndex])
    return ut.removeZerorows(result)

class Config(NamedTuple):
    rvWallThickness: int # RV wall thickness, in [mm] (don't have contours); e.g. 3 ==> downsample by taking every third point
    downsample: int
    upperBdNumContourPts: int # An upper bound on the number of contour points.
    PLOT: bool
    LOAD_MATLAB_VARS: bool

main()
