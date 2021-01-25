from glob import glob
import nibabel as nib
import masks2ContoursMainUtil as mut
import numpy as np
import masks2ContoursUtil as ut

import os
from typing import NamedTuple
from masks2Contours import masks2ContoursSA, masks2ContoursLA

def main():
    use_default_filepaths = True

    # Orientation for viewing plots
    az = 214.5268
    el = -56.0884

    # Configure some settings.
    config = Config(rvWallThickness = 3, downsample = 3, upperBdNumContourPts = 200, PLOT = True, LOAD_MATLAB_VARS = False)
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
    (SAContours, SAinserts) = masks2ContoursSA(segName, imgName, resultsDir, frameNum, config)
    LAContours = masks2ContoursLA(LA_names, LA_segs, resultsDir, frameNum, config)

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

    numSlices = SAendoLVContours.shape[2]
    includedSlices = np.linspace(1, numSlices, numSlices)
    includedSlices = ut.deleteHelper(includedSlices, slicesToOmit, axis = 0)

    # Calculate apex using "method 2" from the MATLAB script.
    LAepiLVContours = LAContours["epiLV"]
    epiPts1 = np.squeeze(LAepiLVContours[:, :, 0])
    epiPts2 = np.squeeze(LAepiLVContours[:, :, 2])
    apex = mut.calcApex(epiPts1, epiPts2)





class Config(NamedTuple):
    rvWallThickness: int # RV wall thickness, in [mm] (don't have contours); e.g. 3 ==> downsample by taking every third point
    downsample: int
    upperBdNumContourPts: int # An upper bound on the number of contour points.
    PLOT: bool
    LOAD_MATLAB_VARS: bool

# [Load results by returning from function]

# This function displays "prompt" to the user and waits for the user to enter input. As long as the user enters
# incorrect input, "errorMessage" and "prompt" are displayed.
#
# The argument "goodInput" must be a boolean function accepting one argument. The intention is that, given a string
# variable called input, goodInput(input) is True when input is valid and false otherwise.
def getCorrectInput(goodInput, prompt, errorMessage):
    validInput = False
    while not validInput:
        returnThis = input(prompt)
        validInput = goodInput(returnThis)
        if not validInput:
            print(errorMessage)

    return returnThis

main()
